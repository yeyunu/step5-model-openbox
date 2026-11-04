---
title: A5 Step5-preview 开箱冒烟验证全过程
date: 2026-10-15
tags:
  - vllm
  - vllm-ascend
  - ascend950
  - step5
---

# A5 Step5-preview 开箱冒烟验证全过程

## 环境与路径

- A5 主机，容器 `yyy-cann92`
- vllm 源码（water18 分支）：`/home/yyy/vllm-ascend-step4`
- vllm-ascend 源码（main 分支）：`/home/yyy/vllm-ascend-main`
- 模型目录：`/home/yyy/weight`（当前只有 `config.json`，无 safetensors）
- 设备：Ascend950DT × 8，每卡 HBM 96GB
- 网络：A5 能连 pypi / gitcode / crates.io，仅 github、gitee 不通；vllm 用 `empty` 目标可完全绕开外网拉取

## 第一阶段：编译

### vllm（water18）

```bash
cd /home/yyy/vllm-ascend-step4
pip install -r requirements/common.txt
VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation
```

- 产物：`vllm 0.1.dev19473+g5d7f9124d.d20261014.empty`
- `empty` 目标下 `ext_modules=[]`，不走 CMake，避免 aarch64 上从 github 拉 ARM Compute Library / oneDNN
- `requirements/common.txt` 会把 fastapi 升到 0.136.x，与 vllm-ascend 的 `fastapi<0.124` pin 冲突（当前共存可用，后续如出问题需拆 venv）

### vllm-ascend（main）

```bash
cd /home/yyy/vllm-ascend-main
sed -i -e 's/^torch-npu/#&/' -e 's/^triton-ascend/#&/' requirements.txt
pip install -r requirements.txt
FLA_NPU_DISABLE_PTH=1 pip install -v -e . --no-build-isolation --no-deps
```

- 产物：`vllm_ascend 0.19.1rc2.dev1939+g80d265f13`，custom ops 全部编译完成
- 关键开关 `FLA_NPU_DISABLE_PTH=1`：镜像里 `fla_npu` 的 `.pth` 会把 12 输入的 CausalConv1d 注入 op store 最前面，盖住 fork 自己的 8 输入版本，导致 opc 编译报 `invalid input nums[8], should be 12`；禁掉注入后编译通过

## 第二阶段：冒烟阻塞点（逐个解决）

1. **`cann_ops_transformer` 缺失**
   - 由 `ops-transformer_inedex_topk` 编译而来，且 `indexer_region_score` 只在 `stepx` 分支
   - 仓库切到 `stepx` 分支，编译 3 个 op 的内核包并安装：

   ```bash
   cd /home/yyy/ops-transformer_inedex_topk
   git checkout -B stepx origin/stepx
   bash build.sh --pkg --soc=ascend950 --vendor_name=step5_ops \
     --ops=generic_block_sparse_attention,generic_block_sparse_attention_metadata,indexer_region_score -j16
   bash build.sh --torch_extension
   ./build_out/cann-ops-transformer-step5_ops_linux-aarch64.run --quiet --install-path=/usr/local/Ascend/cann-9.2.0/opp
   pip install --force-reinstall --no-deps build_out/cann_ops_transformer-1.0.0-py3-none-any.whl
   ```

   - CANN 内旧拷贝备份为 `cann_ops_transformer.bak-20261015` 与对应 `dist-info`

2. **`/dev/shm` 只有 64MB（vllm 要 160MB）**
   - 每个 `docker exec` 是独立 mount namespace，remount 必须写在启动脚本里

   ```bash
   mount -o remount,size=8G /dev/shm
   ```

   - 容器重启后需重做

3. **tokenizer 缺失**
   - 用 HF 缓存里 Qwen3-32B 的 tokenizer 当占位，注入 6 个 step5 特殊 token
   - `<im_patch>`=151669、`<im_start>`=151670、`<im_end>`=151671、`<patch_start>`=151672、`<patch_end>`=151673、`<patch_newline>`=151674
   - 冒烟 config 的 `image_token_id` 改为 151669、`text_config.vocab_size` 改为 152064

4. **这版 vllm 的 `LLM.generate` 无 `prompt_token_ids` 参数**

   ```python
   from vllm.inputs import TokensPrompt
   llm.generate(TokensPrompt(prompt_token_ids=[1, 2, 3, 4, 5]), sampling_params=...)
   ```

5. **孤儿 worker 泄漏显存**
   - 两次失败运行留下 VLLM::EngineCore / Worker 孤儿进程，每批吃 20GB/卡
   - 按 pid 清理后 0-3 卡恢复 91GB/卡空闲

## 减层方案

- 派生 config `/tmp/step5-smoke/config.json`（不动原始权重、不改两个源码目录）
- 92 → 8 层；MTP 3 → 0；`layer_types` / `swiglu_limits` 等按 8 层截断；MoE 层收窄到 3-7；去掉 fp8 `quantization_config`（避免 dummy + fp8 冲突）
- 原始 config：`Step5ForConditionalGeneration`，text_config 为 step5，92 层、352 experts、fp8、MTP 3 层、VLM 带 vision tower

## 冒烟结果

- **TP=8（8 卡）冒烟通过**：`STEP5_SMOKE_OK`，输出 `TOKEN_IDS: [9656, 9656, 9656, 9656, 9656, 9656, 9656, 9656]`，日志 `/tmp/step5_smoke_tp8.log`（2026-10-15 09:21）
- 完整链路：架构检查 → 8 层派生 config → VLM 处理器初始化 → dummy bf16 → profile 前向 → 8 token 解码

## 关键文件与改动清单

| 项 | 位置 |
| --- | --- |
| 冒烟脚本 | `/tmp/step5_smoke_v2.py` |
| 启动脚本（含 shm remount） | `/tmp/run-step5-smoke-tp8.sh` |
| 减层 config + tokenizer | `/tmp/step5-smoke/` |
| ops 内核包 | `/usr/local/Ascend/cann-9.2.0/opp/vendors/step5_ops_transformer` |
| ops Python wheel | site-packages `cann_ops_transformer 1.0.0` |
| 编译日志 | `/tmp/ascend-build92c.log`、`/tmp/step5-ops-build.log` |

环境变量：`PYTHONPATH` 指向两个源码目录、`ASCEND_RT_VISIBLE_DEVICES`、`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`VLLM_STEP4_ENABLE_FA3=0`、`FLA_NPU_DISABLE_PTH=1`，并 source `step5_ops_transformer/bin/set_env.bash`。

## 当前状态与待办

- 已杀掉 `lch-vllm` 容器里占 4-7 卡的 Qwen3.8-27B serve（host PID 2663085 及其 EngineCore/4 个 worker，SIGTERM 正常退出），8 卡全部空闲（91GB/卡）
- TP=8 冒烟已通过；进程退出干净，无孤儿 worker
- 待办：
  - 真实权重放 `/home/yyy/weight/` 后换 `load_format` 做真实验证
  - 容器重启后重做 shm remount；`step5_ops_transformer` 内核包与 wheel 在磁盘上持久
