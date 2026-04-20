from fastapi import APIRouter
from schema import ChatRequest
from fastapi.responses import StreamingResponse
from graph import run_graph

router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        run_graph(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
