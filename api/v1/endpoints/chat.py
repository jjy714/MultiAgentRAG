
from fastapi import APIRouter
from schema import ChatRequest
from fastapi.responses import StreamingResponse
from graph import run_graph
router = APIRouter()

## Chat endpoint that streams SSE events from the graph for the given request
@router.post("/chat")
async def chat(request: ChatRequest):
    """
    args   : {
        "request (ChatRequest)": "incoming chat request with user message and history"
    }
    return : {
        "StreamingResponse": "SSE stream of graph execution events"
    }
    """
    return StreamingResponse(
        run_graph(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
