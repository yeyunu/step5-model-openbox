#!/usr/bin/env python3
"""Generate a readable Markdown README from model-rl-case-guard JSON evidence.

The generated document is intentionally plain Markdown: it renders well in
GitHub, GitLab, VS Code and the Codex file preview without a PDF font/runtime.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Iterable


ENDPOINTS = ("/v1/completions", "/v1/chat/completions")
STATUS_LABELS = {
    "passed": ("✅", "passed"),
    "passed_http_only": ("🟡", "passed (HTTP only)"),
    "failed": ("❌", "failed"),
    "blocked": ("⛔", "blocked"),
    "not_run": ("⏭️", "not run"),
}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def collect_reports(report_dir: Path | None, explicit: Iterable[Path]) -> list[tuple[Path, dict[str, Any]]]:
    paths = list(explicit)
    if report_dir:
        paths.extend(sorted(report_dir.rglob("*.json")))
    reports: list[tuple[Path, dict[str, Any]]] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path in seen or path.name in {"manifest.json", "environment.json"}:
            continue
        seen.add(path)
        data = _read_json(path)
        if data is not None:
            reports.append((path, data))
    return reports


def _status(value: Any) -> tuple[str, str]:
    return STATUS_LABELS.get(str(value or "unknown").lower(), ("❔", str(value or "unknown")))


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("\r", " ").replace("\n", " ").strip()
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _inline(value: Any) -> str:
    return _text(value).replace("`", "\\`").replace("|", "\\|")


def _code(value: Any) -> str:
    return str(value or "").replace("```", "``\\`")


def _output_pairs(data: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    if "weight_update_acceptance" in data:
        before = data.get("baseline_by_endpoint", {})
        after = data.get("post_update_by_endpoint", {})
    else:
        before = data.get("baseline_outputs_by_endpoint") or data.get("baseline_by_endpoint")
        after = data.get("feature_outputs_by_endpoint") or data.get("post_update_by_endpoint")
    if not isinstance(before, dict) and not isinstance(after, dict):
        before = data.get("baseline_outputs") or data.get("baseline") or []
        after = data.get("feature_outputs") or data.get("post_update") or []
        prompts = data.get("prompts") or []
        count = max(len(before) if isinstance(before, list) else 0,
                    len(after) if isinstance(after, list) else 0)
        return [
            (_text(prompts[i] if i < len(prompts) else f"prompt-{i + 1}"),
             before[i] if isinstance(before, list) and i < len(before) else "<missing>",
             after[i] if isinstance(after, list) and i < len(after) else "<missing>")
            for i in range(count)
        ]
    pairs: list[tuple[str, Any, Any]] = []
    prompts = data.get("prompts") or []
    for endpoint in ENDPOINTS:
        left = before.get(endpoint, []) if isinstance(before, dict) else []
        right = after.get(endpoint, []) if isinstance(after, dict) else []
        count = max(len(left) if isinstance(left, list) else 0,
                    len(right) if isinstance(right, list) else 0)
        for i in range(count):
            prompt = prompts[i] if i < len(prompts) else f"prompt-{i + 1}"
            pairs.append((f"{endpoint} · {_text(prompt)}",
                          left[i] if isinstance(left, list) and i < len(left) else "<missing>",
                          right[i] if isinstance(right, list) and i < len(right) else "<missing>"))
    return pairs


def _details(lines: list[str], title: str, value: Any) -> None:
    if value:
        lines.extend(["", "<details>", f"<summary>{title}</summary>", "", "```text", _code(value), "```", "", "</details>"])


def _relative_path(path: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(path, output_dir)).as_posix()


def build_readme(
    reports: list[tuple[Path, dict[str, Any]]],
    commands: str,
    *,
    title: str,
    model: str | None,
    output_dir: Path,
) -> str:
    statuses = [_status(data.get("status")) for _, data in reports]
    passed = sum(label.startswith("passed") for _, label in statuses)
    lines = [f"# {title}", "", "> 自动汇总的模型 RL 用例看护证据。原始日志、JSON 报告和启动命令应与本文件一起归档。", ""]
    lines.extend(["## Summary", "", f"- **Generated:** {dt.datetime.now().astimezone().isoformat(timespec='seconds')}"])
    if model:
        lines.append(f"- **Model / checkpoint:** `{_inline(model)}`")
    lines.append(f"- **Result:** **{passed}/{len(statuses)}** report(s) passed" if statuses else "- **Result:** no JSON reports found")
    lines.extend(["", "| Feature | Status | Evidence |", "|---|---|---|"])
    if not reports:
        lines.append("| — | ❔ unknown | No JSON report found |")
    for path, data in reports:
        icon, label = _status(data.get("status"))
        feature = data.get("feature") or data.get("mode") or path.stem
        relative = _relative_path(path, output_dir)
        lines.append(f"| `{_inline(feature)}` | {icon} **{label}** | [`{relative}`]({relative}) |")

    if commands.strip():
        lines.extend(["", "## Commands", "", "The following commands were recorded in `logs/launch_commands.txt`:", "", "```bash", _code(commands.rstrip()), "```"])

    lines.extend(["", "## Feature evidence"])
    for index, (path, data) in enumerate(reports, 1):
        icon, label = _status(data.get("status"))
        feature = data.get("feature") or data.get("mode") or path.stem
        relative = _relative_path(path, output_dir)
        lines.extend(["", f"### {index}. {icon} {feature} — {label}", "", f"Evidence file: [`{relative}`]({relative})"])
        checks = data.get("checks")
        if isinstance(checks, dict):
            lines.extend(["", "#### Checks", "", "| Check | Result |", "|---|---|"])
            for name, value in checks.items():
                result = "✅ pass" if value is True else ("❌ fail" if value is False else f"`{_inline(value)}`")
                lines.append(f"| `{_inline(name)}` | {result} |")
        endpoint_checks = data.get("endpoint_checks")
        if isinstance(endpoint_checks, dict):
            lines.extend(["", "#### Endpoint checks", "", "| Endpoint | Result |", "|---|---|"])
            for endpoint, value in endpoint_checks.items():
                lines.append(f"| `{_inline(endpoint)}` | {'✅ pass' if value is True else '❌ fail'} |")
        for key, heading in (("checkpoint", "Checkpoint"), ("service_url", "Service"), ("log", "Log"), ("log_path", "Log path"), ("server_log", "Server log"), ("client_log", "Client log")):
            if data.get(key):
                lines.append(f"- **{heading}:** `{_inline(data[key])}`")
        for key, heading in (("reason", "Reason"), ("error", "Error"), ("recovery", "Recovery"), ("note", "Note")):
            if data.get(key):
                lines.append(f"- **{heading}:** {_inline(data[key])}")
        _details(lines, "Launch command", data.get("launch_command"))
        _details(lines, "Verification command", data.get("verification_command"))
        pairs = _output_pairs(data)
        if pairs:
            lines.extend(["", "#### Output comparison", "", "| Endpoint / prompt | Baseline | After update |", "|---|---|---|"])
            for prompt, before, after in pairs:
                lines.append(f"| {_inline(prompt)} | {_inline(before)} | {_inline(after)} |")
        elif data.get("outputs"):
            lines.extend(["", "#### Outputs", ""])
            lines.extend(f"- {_inline(item)}" for item in data["outputs"])

    lines.extend(["", "## Interpretation", "", "- `passed` means the recorded checks completed; it is not a substitute for customer precision acceptance.", "- `failed`, `blocked` and `not_run` must be reviewed together with the original logs and environment evidence.", "- Text output is runtime evidence only; it does not prove numerical equivalence.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Markdown README from model-rl-case-guard JSON evidence")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, action="append", default=[])
    parser.add_argument("--commands-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="模型 RL 用例看护报告")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)
    commands = args.commands_file.read_text(encoding="utf-8", errors="replace") if args.commands_file and args.commands_file.is_file() else ""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = build_readme(
        collect_reports(args.report_dir, args.report),
        commands,
        title=args.title,
        model=args.model,
        output_dir=args.output.parent.resolve(),
    )
    args.output.write_text(content + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
