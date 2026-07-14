"""
conftest.py — pytest 전역 설정

watermark 패키지를 sys.path에 추가해 tests/ 에서 직접 import 가능하게 함.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 — tests/에서 watermark.* import 가능
sys.path.insert(0, str(Path(__file__).parent))
