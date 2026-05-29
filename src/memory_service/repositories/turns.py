from __future__ import annotations
import json
from datetime import datetime
from psycopg_pool import ConnectionPool
from ..models.domain import Turn


class TurnRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def insert(self, turn: Turn) -> str:
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO turns (id, session_id, user_id, messages, timestamp, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (turn.id, turn.session_id, turn.user_id, json.dumps(turn.messages),
                 turn.timestamp, json.dumps(turn.metadata)),
            )
            conn.commit()
        return turn.id

    def get(self, turn_id: str) -> Turn | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, session_id, user_id, messages, timestamp, metadata FROM turns WHERE id=%s",
                (turn_id,),
            ).fetchone()
        return self._row(row) if row else None

    def recent_for_session(self, session_id: str, limit: int = 5) -> list[Turn]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT id, session_id, user_id, messages, timestamp, metadata
                   FROM turns WHERE session_id=%s ORDER BY timestamp DESC, created_at DESC LIMIT %s""",
                (session_id, limit),
            ).fetchall()
        return [self._row(r) for r in rows]

    def delete_session(self, session_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM turns WHERE session_id=%s", (session_id,))
            conn.commit()

    def delete_user(self, user_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM turns WHERE user_id=%s", (user_id,))
            conn.commit()

    @staticmethod
    def _row(r) -> Turn:
        msgs = r[3] if isinstance(r[3], list) else json.loads(r[3])
        meta = r[5] if isinstance(r[5], dict) else json.loads(r[5])
        return Turn(id=r[0], session_id=r[1], user_id=r[2], messages=msgs,
                    timestamp=r[4], metadata=meta)
