from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    try:
        with request.app.state.pool.connection() as conn:
            conn.execute("SELECT 1")
        return {"status": "ok"}
    except Exception:
        from fastapi.responses import JSONResponse
        return JSONResponse({"status": "degraded"}, status_code=503)
