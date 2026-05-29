from __future__ import annotations
from psycopg_pool import ConnectionPool
from ..models.domain import Memory, MemoryCandidate, MemoryType


class MemoryRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def insert(self, mem_id: str, user_id: str | None, session_id: str,
               cand: MemoryCandidate, source_turn: str,
               embedding: list[float], supersedes: str | None) -> str:
        with self._pool.connection() as conn:
            if supersedes:
                conn.execute("UPDATE memories SET active=FALSE, updated_at=now() WHERE id=%s",
                             (supersedes,))
            conn.execute(
                """INSERT INTO memories
                   (id, user_id, session_id, type, key, value, confidence,
                    source_session, source_turn, supersedes, active, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE,%s)""",
                (mem_id, user_id, session_id, cand.type.value, cand.key, cand.value,
                 cand.confidence, session_id, source_turn, supersedes, embedding),
            )
            conn.commit()
        return mem_id

    def active_by_key(self, user_id: str | None) -> dict[str, tuple[str, str]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT key, id, value FROM memories WHERE user_id IS NOT DISTINCT FROM %s AND active=TRUE",
                (user_id,),
            ).fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}

    def active_facts(self, user_id: str | None) -> list[Memory]:
        return [m for m in self.list_for_user(user_id, include_superseded=False)
                if m.type is MemoryType.fact]

    def search_vector(self, user_id: str | None, session_id: str,
                      embedding: list[float], limit: int = 20) -> list[str]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT id FROM memories
                   WHERE active=TRUE AND (
                       (%s::text IS NOT NULL AND user_id = %s)
                       OR (%s::text IS NULL AND session_id = %s))
                   ORDER BY embedding <=> %s::vector LIMIT %s""",
                (user_id, user_id, user_id, session_id, embedding, limit),
            ).fetchall()
        return [r[0] for r in rows]

    def search_fts(self, user_id: str | None, session_id: str,
                   query: str, limit: int = 20) -> list[str]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT id, ts_rank(fts, plainto_tsquery('english', %s)) AS rank
                   FROM memories
                   WHERE active=TRUE AND (
                       (%s::text IS NOT NULL AND user_id = %s)
                       OR (%s::text IS NULL AND session_id = %s))
                     AND fts @@ plainto_tsquery('english', %s)
                   ORDER BY rank DESC LIMIT %s""",
                (query, user_id, user_id, user_id, session_id, query, limit),
            ).fetchall()
        return [r[0] for r in rows]

    def get_many(self, ids: list[str]) -> dict[str, Memory]:
        if not ids:
            return {}
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""SELECT {self._cols()} FROM memories WHERE id = ANY(%s)""",
                (ids,),
            ).fetchall()
        return {r[0]: self._row(r) for r in rows}

    def list_for_user(self, user_id: str | None, include_superseded: bool = True) -> list[Memory]:
        clause = "" if include_superseded else " AND active=TRUE"
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""SELECT {self._cols()} FROM memories
                    WHERE user_id IS NOT DISTINCT FROM %s{clause}
                    ORDER BY created_at""",
                (user_id,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def delete_session(self, session_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("UPDATE memories SET supersedes=NULL WHERE supersedes IN "
                         "(SELECT id FROM memories WHERE session_id=%s)", (session_id,))
            conn.execute("DELETE FROM memories WHERE session_id=%s", (session_id,))
            conn.commit()

    def delete_user(self, user_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("UPDATE memories SET supersedes=NULL WHERE supersedes IN "
                         "(SELECT id FROM memories WHERE user_id=%s)", (user_id,))
            conn.execute("DELETE FROM memories WHERE user_id=%s", (user_id,))
            conn.commit()

    @staticmethod
    def _cols() -> str:
        return ("id, user_id, session_id, type, key, value, confidence, source_session, "
                "source_turn, created_at, updated_at, supersedes, active")

    @staticmethod
    def _row(r) -> Memory:
        return Memory(id=r[0], user_id=r[1], session_id=r[2], type=MemoryType(r[3]),
                      key=r[4], value=r[5], confidence=r[6], source_session=r[7],
                      source_turn=r[8], created_at=r[9], updated_at=r[10],
                      supersedes=r[11], active=r[12])
