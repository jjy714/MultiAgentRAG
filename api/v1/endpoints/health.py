from fastapi import APIRouter

router = APIRouter()
## Health check endpoint to verify the API is running
@router.get("/health")
async def health():
    """
    args   : {}
    return : {
        "dict": "status and service name"
    }
    """
    return {"status": "ok", "service": "MultiAgentRAG API"}
