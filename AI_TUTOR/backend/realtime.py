"""Milestone 1: realtime session endpoints (browser <-> LiveKit).

The browser asks this backend for a short-lived JWT; LiveKit API secrets
never leave the server. Later milestones attach the Pipecat pipeline to the
same room instead of a second human participant.
"""
import os

from dotenv import load_dotenv
from fastapi import APIRouter
from livekit import api as lk_api
from pydantic import BaseModel, Field

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")

router = APIRouter(prefix="/api/session", tags=["session"])


class TokenRequest(BaseModel):
    identity: str = Field(min_length=1)
    room: str = "tutor-room"


class TokenResponse(BaseModel):
    url: str
    token: str


@router.post("/token", response_model=TokenResponse)
def create_session_token(req: TokenRequest):
    token = (
        lk_api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(req.identity)
        .with_name(req.identity)
        .with_grants(lk_api.VideoGrants(room_join=True, room=req.room))
        .to_jwt()
    )
    return TokenResponse(url=LIVEKIT_URL, token=token)
