"""
Call Center Transcription Service
Hosted on Railway - receives recording URL, returns transcript via OpenRouter
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import ssl
import base64
import os
from typing import Optional

app = FastAPI(title="Call Center Transcriber")

# SSL context for ViciDial (self-signed cert)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# OpenRouter API key from environment or .env file
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_KEY:
    # Try loading from .env file
    try:
        with open('.env') as f:
            for line in f:
                if line.startswith('OPENROUTER_API_KEY='):
                    OPENROUTER_KEY = line.split('=', 1)[1].strip()
                    break
    except:
        pass

if not OPENROUTER_KEY:
    print("WARNING: OPENROUTER_API_KEY not set - transcription will fail")


class TranscribeRequest(BaseModel):
    recording_url: str
    phone: Optional[str] = None


class TranscribeResponse(BaseModel):
    transcript: str
    phone: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest):
    """
    Download recording from ViciDial and transcribe via OpenRouter gpt-4o-audio-preview
    """
    import time
    start_time = time.time()

    try:
        # Download audio with SSL bypass for ViciDial
        async with httpx.AsyncClient(verify=False, timeout=120.0) as client:
            audio_response = await client.get(req.recording_url)
            audio_response.raise_for_status()
            audio_bytes = audio_response.content

        # Encode to base64
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Call OpenRouter
        payload = {
            "model": "openai/gpt-4o-audio-preview",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this solar appointment call verbatim."},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}}
                ]
            }]
        }

        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://call-center.magneticexteriors.com",
                    "X-Title": "Call Center Transcriber"
                },
                json=payload
            )
            response.raise_for_status()
            result = response.json()

        transcript = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        return TranscribeResponse(
            transcript=transcript,
            phone=req.phone,
            duration_seconds=round(time.time() - start_time, 1)
        )

    except Exception as e:
        return TranscribeResponse(
            transcript="",
            phone=req.phone,
            duration_seconds=round(time.time() - start_time, 1) if 'start_time' in dir() else None,
            error=str(e)
        )


# No __main__ needed - Railpack runs uvicorn via start command
