"""
Call Center Transcription Service
Hosted on Railway - uses OpenRouter whisper-1 for transcription
Accepts either recording_url (downloads + transcribes) or audio_base64 (transcribes only)
Cost: ~$0.03 per 5-minute call
"""
from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import base64
import os
import time
from typing import Optional

app = FastAPI(title="Call Center Transcriber")

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")


class TranscribeRequest(BaseModel):
    recording_url: Optional[str] = None
    audio_base64: Optional[str] = None
    audio_format: str = "mp3"
    phone: Optional[str] = None


class TranscribeResponse(BaseModel):
    transcript: str
    phone: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    cost: Optional[float] = None


@app.get("/health")
def health():
    return {"status": "ok", "model": "openai/whisper-1"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest):
    start_time = time.time()

    try:
        audio_b64 = req.audio_base64

        # If URL provided, download first
        if not audio_b64 and req.recording_url:
            async with httpx.AsyncClient(verify=False, timeout=120.0) as client:
                audio_response = await client.get(req.recording_url)
                audio_response.raise_for_status()
                audio_b64 = base64.b64encode(audio_response.content).decode('utf-8')

        if not audio_b64:
            return TranscribeResponse(
                transcript="",
                phone=req.phone,
                error="No audio provided (need recording_url or audio_base64)"
            )

        # Call OpenRouter whisper-1
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/audio/transcriptions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/whisper-1",
                    "input_audio": {
                        "data": audio_b64,
                        "format": req.audio_format
                    }
                }
            )
            response.raise_for_status()
            result = response.json()

        transcript = result.get("text", "")
        usage = result.get("usage", {})
        cost = usage.get("cost")

        return TranscribeResponse(
            transcript=transcript,
            phone=req.phone,
            duration_seconds=round(time.time() - start_time, 1),
            cost=cost
        )

    except Exception as e:
        return TranscribeResponse(
            transcript="",
            phone=req.phone,
            duration_seconds=round(time.time() - start_time, 1),
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
