"""从日志流中提取早期底层错误及原始堆栈，不被 torchrun 末尾汇总遮挡。"""

import re
import sys
from collections import deque
from pathlib import Path


def summarize(path):
    important = []
    traceback = []
    collecting = False
    tail = deque(maxlen=25)
    pattern = re.compile(
        r"NCCL (?:WARN|ERROR)|Hggc failure|CUDA out of memory|"
        r"(?:DistBackendError|RuntimeError|OutOfMemoryError|ValueError|AttributeError|ImportError|TypeError|FileNotFoundError):"
    )
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.rstrip()
            tail.append(line)
            if pattern.search(line) and len(important) < 16:
                important.append(f"L{number}: {line}")
            if "Traceback (most recent call last)" in line and not traceback:
                collecting = True
            if collecting and len(traceback) < 35:
                traceback.append(f"L{number}: {line}")
                if re.search(r"(?:Error|Exception):", line):
                    collecting = False
    if important:
        print("=== Earliest backend / exception messages ===")
        print("\n".join(important))
    if traceback:
        print("=== First original traceback ===")
        print("\n".join(traceback))
    print("=== Last 25 lines ===")
    print("\n".join(tail))


if __name__ == "__main__":
    summarize(sys.argv[1])
