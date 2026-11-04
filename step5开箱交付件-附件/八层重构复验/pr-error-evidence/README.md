# 上游修复对应的本项目日志证据

## 检索范围与证据口径

只读检索 225、226 上原测试容器对应的宿主机绑定目录；容器处于停止状态，未启动容器、模型或新测试。扫描测试目录下的 `.log`、`.txt`、`.out` 文本，跳过源码、环境、权重及备份目录，单文件上限 30 MB。另核对已有的 A5 回归日志归档和本机 CPU 回归日志。

- 225：`/data/yyy/hccl-without-pr19248-20261030/home/model_openbox`。
- 226：`/data/yyy/upstream-retest-20261030/home/model_openbox`。
- 原始路径与 SHA256：[manifest.json](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/manifest.json)。附件保持原始内容和时间记录。

## 对应关系

| 修复 | 本项目证据 | 能支持的结论 |
| --- | --- | --- |
| #16959 IPC reload 生命周期 | [集成回归](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/ascend-integration-cpu.log)，61 passed | 本次检索未找到可直接归因于未进入 reload 或 packed 加载遗漏的修复前运行时报错；集成通过不能替代失败复现。 |
| #17154 HCCL 通信器销毁 | [集成回归](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/ascend-integration-cpu.log)，61 passed | 未找到本项目 native 通信器未销毁的直接报错或泄漏测量；不能用进程退出日志证明此因果。 |
| #18848 IPC 基类复用 | [集成回归](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/ascend-integration-cpu.log)，61 passed | 未找到本项目错误 UUID/handle 或分块 MTP 检查失败的直接报错；当前证据为集成回归。 |
| #18944 HCCL 基类复用 | [集成回归](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/ascend-integration-cpu.log)，61 passed | 未找到本项目 packed 配置冲突、旧 trainer 状态机或异常清理缺口的直接报错；当前证据为集成回归。 |
| #20740 / #51259 NPU 导入缓存 | 225 两轮、226 两轮客户端日志各出现 8 条共享 NPU 张量未释放告警；本机修复前 CPU 回归两项在 `assert 2 == 1` 失败 | 告警证明出现了共享张量生命周期异常现象；CPU 负向对照直接证明同一导出重复导入。告警本身不证明引用计数具体数值或每轮泄漏字节数。 |
| #20403 专家映射 | A5 已归档失败回归：映射被替换成 meta tensor；计数诊断：修复前 32/384、保留 2 个输入张量，修复后 32/32、保留 0 个；修复后 5 passed | 直接支持专家映射存储丢失、完成计数错误及模块接收缓存不释放的因果。该证据为 CPU 回归/诊断，不是 NPU OOM 日志。 |

同一份 61 项集成回归是组合验证记录，不能拆写成每个 PR 各自有 61 项测试通过。未找到表示在本次检索范围内未发现，不代表问题从未出现。

## 原始日志摘录

### 225-ipc-before-round1

原始日志：[225-ipc-before-round1.log](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/225-ipc-before-round1.log)

```text
L217: [W1102 15:37:16.331590920 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L218: [W1102 15:37:16.389533640 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L219: [W1102 15:37:17.350818700 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L220: [W1102 15:37:17.356555310 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L221: [W1102 15:37:17.356705970 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L222: [W1102 15:37:17.387740450 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L223: [W1102 15:37:18.485682770 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L224: [W1102 15:37:18.903520050 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
```

### 225-ipc-before-round2

原始日志：[225-ipc-before-round2.log](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/225-ipc-before-round2.log)

```text
L217: [W1102 15:38:37.213876920 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L218: [W1102 15:38:37.219176270 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L219: [W1102 15:38:37.231878700 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L220: [W1102 15:38:38.934152660 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L221: [W1102 15:38:38.314337950 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L222: [W1102 15:38:38.317353250 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L223: [W1102 15:38:39.441272150 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L224: [W1102 15:38:39.627880280 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
```

### 226-ipc-before

原始日志：[226-ipc-before.log](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/226-ipc-before.log)

```text
L226: [W1030 05:32:59.446693430 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L227: [W1030 05:32:59.647342010 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L228: [W1030 05:32:59.666567980 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L229: [W1030 05:32:59.676725810 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L230: [W1030 05:32:59.172315010 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L231: [W1030 05:33:00.163749560 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L232: [W1030 05:33:01.474867830 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L233: [W1030 05:33:01.681858680 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
```

### 226-ipc-before-repeat

原始日志：[226-ipc-before-repeat.log](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/226-ipc-before-repeat.log)

```text
L226: [W1030 05:35:44.923612210 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L227: [W1030 05:35:44.026474330 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L228: [W1030 05:35:44.043900250 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L229: [W1030 05:35:45.386988230 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L230: [W1030 05:35:45.404835850 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L231: [W1030 05:35:45.407071030 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L232: [W1030 05:35:45.150623090 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
L233: [W1030 05:35:47.670979710 NPUIPCTypes.cpp:23] Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]
```

### 20403-regression-before

原始日志：[20403-regression-before.log](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/20403-regression-before.log)

```text
L15: E   AssertionError: assert tensor(..., device='meta', size=(352,), dtype=torch.int32) is tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,\n        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,\n        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,\n        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,\n        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,  0,  1,\n      
L16: E    +  where tensor(..., device='meta', size=(352,), dtype=torch.int32) = Module().expert_ids_per_ep_rank
L37: ================= 1 failed, 4 deselected, 17 warnings in 0.12s =================
```

### 20403-regression-after

原始日志：[20403-regression-after.log](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/20403-regression-after.log)

```text
L33: ======================== 5 passed, 17 warnings in 0.05s ========================
```

### 20403-count-diagnostic

原始日志：[20403-count-diagnostic.log](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/20403-count-diagnostic.log)

```text
L12: {"mode": "without_fix", "real_weight_numel_received": 32, "expected_numel": 384, "retained_input_tensors": 2, "weight_still_meta": true, "same_expert_map": false}
L13: {"mode": "preserve_map_only", "real_weight_numel_received": 32, "expected_numel": 384, "retained_input_tensors": 2, "weight_still_meta": true, "same_expert_map": true}
L14: {"mode": "minimal_fix", "real_weight_numel_received": 32, "expected_numel": 32, "retained_input_tensors": 0, "weight_still_meta": false, "same_expert_map": true}
```

### ascend-integration-cpu

原始日志：[ascend-integration-cpu.log](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/ascend-integration-cpu.log)

```text
L18: 61 passed, 15 warnings in 0.61s
```

### ipc-import-before-regression

原始日志：[ipc-import-before-regression.log](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/ipc-import-before-regression.log)

```text
L45: E           assert 2 == 1
L91: E           assert 2 == 1
L98: 2 failed, 17 deselected in 0.10s
```
