from fastapi import APIRouter, Request
from ..models.schemas import MemoriesResponse, MemoryOut

router = APIRouter()


@router.get("/users/{user_id}/memories", response_model=MemoriesResponse)
def get_memories(user_id: str, request: Request, include_superseded: bool = True):
    rows = request.app.state.memories.list_for_user(user_id, include_superseded=include_superseded)
    out = [MemoryOut(id=m.id, type=m.type.value, key=m.key, value=m.value,
                     confidence=m.confidence, source_session=m.source_session,
                     source_turn=m.source_turn, created_at=m.created_at.isoformat(),
                     updated_at=m.updated_at.isoformat(), supersedes=m.supersedes,
                     active=m.active) for m in rows]
    return MemoriesResponse(memories=out)
