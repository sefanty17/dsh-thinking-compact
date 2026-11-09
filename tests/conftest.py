import sys
from pathlib import Path

# 让 `python -m pytest tests` 在仓库根目录直接可以 import 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
