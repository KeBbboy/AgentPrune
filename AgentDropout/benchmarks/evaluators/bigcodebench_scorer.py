"""BigCodeBench 评测器。

BigCodeBench 的答案不是字符串精确匹配, 而是把模型生成的函数实现和
数据集提供的 unittest 测试拼在一起执行, 通过测试即为正确。
"""

import os
import re
import signal
import subprocess
import sys
import tempfile
from typing import Any, Dict


def _strip_code_fences(text: Any) -> str:
    """提取 markdown 代码块中的 Python 代码, 并移除工具观测前缀。"""

    raw = "" if text is None else str(text).strip()
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[-1].strip()

    matches = list(re.finditer(r"```(?:python)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL))
    if matches:
        raw = matches[-1].group(1).strip()

    raw = re.sub(r"^\s*\[Syntax check:\s*OK\]\s*", "", raw, flags=re.IGNORECASE)
    return raw.strip()


def evaluate_bigcodebench(
    generated_code: str,
    metadata: Dict[str, Any],
    timeout: int = 30,
) -> Dict[str, Any]:
    """运行 BigCodeBench 的 unittest 测试片段并返回 pass/fail。"""

    task_id = str(metadata.get("task_id", "unknown"))
    code_prompt = str(metadata.get("code_prompt", ""))
    test_harness = str(metadata.get("test", ""))
    entry_point = str(metadata.get("entry_point", ""))

    impl = _strip_code_fences(generated_code)
    if not impl:
        return {
            "correct": False,
            "score": 0.0,
            "accuracy": 0.0,
            "task_id": task_id,
            "prediction": "",
            "error": "Empty generated code",
        }

    if entry_point and re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", impl, re.MULTILINE):
        full_code = impl
    else:
        full_code = code_prompt.rstrip() + "\n" + impl

    assembled = (
        full_code.rstrip()
        + "\n\n"
        + test_harness.strip()
        + "\n\n"
        + "import unittest\n"
        + "unittest.main(argv=[''], exit=True, verbosity=0)\n"
    )

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(assembled)
            tmp_path = tmp.name

        process = subprocess.Popen(
            [sys.executable, tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            close_fds=True,
        )
        try:
            _, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                if hasattr(os, "setsid"):
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=5)
            except Exception:
                pass
            return {
                "correct": False,
                "score": 0.0,
                "accuracy": 0.0,
                "task_id": task_id,
                "prediction": impl,
                "error": f"Timeout after {timeout}s",
            }

        if process.returncode == 0:
            return {
                "correct": True,
                "score": 1.0,
                "accuracy": 1.0,
                "task_id": task_id,
                "prediction": impl,
                "error": None,
            }

        error_lines = [line.strip() for line in stderr.strip().splitlines() if line.strip()]
        error_msg = error_lines[-1][:300] if error_lines else "Unknown error"
        return {
            "correct": False,
            "score": 0.0,
            "accuracy": 0.0,
            "task_id": task_id,
            "prediction": impl,
            "error": error_msg,
        }
    except Exception as exc:
        return {
            "correct": False,
            "score": 0.0,
            "accuracy": 0.0,
            "task_id": task_id,
            "prediction": impl,
            "error": str(exc),
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
