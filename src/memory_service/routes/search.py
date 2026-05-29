from fastapi import APIRouter, Request
from ..models.schemas import SearchRequest, SearchResponse

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def post_search(req: SearchRequest, request: Request):
    hits = request.app.state.recall.search(req.query, req.session_id, req.user_id, req.limit)
    return SearchResponse(results=hits)
