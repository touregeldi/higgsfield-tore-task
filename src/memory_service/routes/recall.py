from fastapi import APIRouter, Request
from ..models.schemas import RecallRequest, RecallResponse

router = APIRouter()


@router.post("/recall", response_model=RecallResponse)
def post_recall(req: RecallRequest, request: Request):
    return request.app.state.recall.recall(req.query, req.session_id, req.user_id, req.max_tokens)
