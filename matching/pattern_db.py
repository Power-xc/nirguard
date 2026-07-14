"""
matching/pattern_db.py — 패턴 데이터베이스

User ID → 64×64 이진 패턴을 SQLite에 저장하고,
복원된 패턴과 해밍 거리를 기반으로 후보 Top-K를 반환한다.

거리·신뢰도 계산 공식은 watermark.confidence 한 곳에서만 정의한다.
"""

import sqlite3
import threading
from pathlib import Path

import numpy as np

from watermark.confidence import confidence_from_hamming
from watermark.generator import PATTERN_SIZE

_PATTERN_BITS = PATTERN_SIZE * PATTERN_SIZE  # 4,096


class PatternDB:
    """
    NIRGuard 유출자 패턴 데이터베이스.

    SQLite 파일 기반 — 데모용 로컬 저장소.
    프로덕션에서는 Redis/Faiss로 교체 가능 (인터페이스 동일).
    """

    def __init__(self, db_path: str = "matching/nirguard.db") -> None:
        """
        Args:
            db_path: SQLite 파일 경로. 부모 디렉토리가 없으면 자동 생성.
        """
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False + Lock: FastAPI 스레드풀에서 안전하게 공유
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        """테이블 초기화 — 없으면 생성."""
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    user_id  TEXT PRIMARY KEY,
                    pattern  BLOB NOT NULL,
                    added_at TEXT DEFAULT (datetime('now'))
                )
            """)
            self._conn.commit()

    def add_pattern(self, user_id: str, pattern: np.ndarray) -> None:
        """
        user_id와 64×64 이진 패턴을 DB에 저장한다.

        이미 존재하면 덮어쓴다 (REPLACE).

        Args:
            user_id: 유출자 식별자 (예: "user_003")
            pattern: generate_pattern()이 반환한 (64, 64) uint8 배열
        """
        if pattern.shape != (PATTERN_SIZE, PATTERN_SIZE):
            raise ValueError(
                f"패턴 shape 오류: 기대 ({PATTERN_SIZE}, {PATTERN_SIZE}), "
                f"실제 {pattern.shape}"
            )
        # uint8 배열 → bytes로 직렬화
        blob: bytes = pattern.astype(np.uint8).tobytes()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO patterns (user_id, pattern) VALUES (?, ?)",
                (user_id, blob),
            )
            self._conn.commit()

    def match(
        self,
        recovered: np.ndarray,
        top_k: int = 3,
    ) -> list[dict]:
        """
        복원된 패턴을 DB 전체와 비교해 유출자 후보 Top-K를 반환한다.

        해밍 거리가 낮을수록 원본 패턴에 가깝다.
        신뢰도 = max(0, 1 - 2 × 해밍거리) × 100%

        Args:
            recovered: extract_from_image/video()가 반환한 (64, 64) uint8 패턴
            top_k:     반환할 후보 수 (기본 3)

        Returns:
            [
              {"user_id": str, "hamming": float, "confidence": float},
              ...
            ]  — 해밍 거리 오름차순 (신뢰도 내림차순)
        """
        if recovered.shape != (PATTERN_SIZE, PATTERN_SIZE):
            raise ValueError(f"패턴 shape 오류: {recovered.shape}")

        recovered_flat = recovered.flatten().astype(np.uint8)

        with self._lock:
            rows = self._conn.execute(
                "SELECT user_id, pattern FROM patterns"
            ).fetchall()

        if not rows:
            return []

        results: list[dict] = []
        for user_id, blob in rows:
            stored = np.frombuffer(blob, dtype=np.uint8)
            if stored.size != _PATTERN_BITS:
                # DB에 비정상 레코드가 있으면 건너뜀
                continue

            errors: int = int(np.sum(recovered_flat ^ stored))
            hamming: float = round(errors / _PATTERN_BITS, 4)
            confidence: float = confidence_from_hamming(hamming)

            results.append({
                "user_id": user_id,
                "hamming": hamming,
                "confidence": confidence,
            })

        # 해밍 거리 오름차순 정렬 → Top-K 반환
        results.sort(key=lambda x: x["hamming"])
        return results[:top_k]

    def count(self) -> int:
        """DB에 저장된 패턴 수를 반환한다."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM patterns").fetchone()
        return int(row[0])

    def has_user(self, user_id: str) -> bool:
        """user_id가 DB에 존재하는지 확인한다."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM patterns WHERE user_id = ?", (user_id,)
            ).fetchone()
        return row is not None

    def get_demo_users(self, n: int = 10) -> list[str]:
        """데모용 user_id 목록을 반환한다."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT user_id FROM patterns ORDER BY user_id LIMIT ?", (n,)
            ).fetchall()
        return [r[0] for r in rows]

    def clear(self) -> None:
        """DB의 모든 패턴을 삭제한다 (테스트/초기화용)."""
        with self._lock:
            self._conn.execute("DELETE FROM patterns")
            self._conn.commit()

    def close(self) -> None:
        """DB 연결을 닫는다."""
        with self._lock:
            self._conn.close()
