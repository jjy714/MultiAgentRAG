from api.v1.endpoints import chat, health
from fastapi import APIRouter


api_router = APIRouter()

api_router.include_router(chat.router, prefix="/api", tags=["evaluate"])
api_router.include_router(health.router, prefix="/api", tags=["evaluate"])