from datetime import datetime, timezone
from fastapi import APIRouter, Request, status
from ..models.schemas import TurnRequest, TurnResponse
from ..models.domain import Turn

router = APIRouter()


@router.post("/turns", response_model=TurnResponse, status_code=status.HTTP_201_CREATED)
def post_turn(req: TurnRequest, request: Request):
    try:
        ts = datetime.fromisoformat(req.timestamp.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        ts = datetime.now(timezone.utc)  # tz-aware fallback for TIMESTAMPTZ column
    turn = Turn(id="", session_id=req.session_id, user_id=req.user_id,
                messages=[m.model_dump() for m in req.messages], timestamp=ts,
                metadata=req.metadata)
    turn_id = request.app.state.ingest.ingest(turn)
    return TurnResponse(id=turn_id)
