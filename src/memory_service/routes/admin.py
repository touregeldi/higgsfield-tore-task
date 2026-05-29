from fastapi import APIRouter, Request, Response, status

router = APIRouter()


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str, request: Request):
    request.app.state.memories.delete_session(session_id)
    request.app.state.turns.delete_session(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: str, request: Request):
    request.app.state.memories.delete_user(user_id)
    request.app.state.turns.delete_user(user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
