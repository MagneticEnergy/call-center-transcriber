"""
Call Center Transcription Service
Hosted on Railway - uses faster-whisper (local Whisper) for FREE transcription
Fallback to OpenRouter gpt-audio-mini if local fails
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import ssl
import base64
import os
import time
from typing import Optional

app = FastAPI(title="Call Center Transcriber")

# SSL context for ViciDial (self-signed cert)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# OpenRouter API key (fallback only)
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Try to import faster-whisper for local transcription (FREE)
whisper_model = None
try:
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    print("faster-whisper loaded: base model (FREE, local)")
except ImportError:
    print("faster-whisper not available, will use OpenRouter fallback")
except Exception as e:
    print(f"faster-whisper load failed: {e}, will use OpenRouter fallback")


class TranscribeRequest(BaseModel):
    recording_url: str
    phone: Optional[str] = None


class TranscribeResponse(BaseModel):
    transcript: str
    phone: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    method: Optional[str] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "whisper_local": whisper_model is not None,
        "openrouter_fallback": bool(OPENROUTER_KEY)
    }


def transcribe_local(audio_bytes: bytes) -> str:
    """Transcribe using local faster-whisper (FREE)"""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        segments, info = whisper_model.transcribe(tmp_path, beam_size=5)
        transcript = " ".join(segment.text for segment in segments).strip()
        return transcript
    finally:
        os.unlink(tmp_path)


async def transcribe_openrouter(audio_bytes: bytes) -> str:
    """Transcribe via OpenRouter gpt-audio-mini (cheap fallback)"""
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

    payload = {
        "model": "openai/gpt-audio-mini",
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

    return result.get("choices", [{}])[0].get("message", {}).get("content", "")


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest):
    start_time = time.time()

    try:
        # Download audio with SSL bypass for ViciDial
        async with httpx.AsyncClient(verify=False, timeout=120.0) as client:
            audio_response = await client.get(req.recording_url)
            audio_response.raise_for_status()
            audio_bytes = audio_response.content

        # Try local Whisper first (FREE)
        if whisper_model:
            try:
                transcript = transcribe_local(audio_bytes)
                method = "whisper-local"
            except Exception as e:
                print(f"Local whisper failed: {e}, falling back to OpenRouter")
                transcript = ""
                method = ""
        
        # Fallback to OpenRouter if local failed or unavailable
        if not whisper_model or not transcript:
            if OPENROUTER_KEY:
                transcript = await transcribe_openrouter(audio_bytes)
                method = "openrouter-gpt-audio-mini"
            else:
                return TranscribeResponse(
                    transcript="",
                    phone=req.phone,
                    duration_seconds=round(time.time() - start_time, 1),
                    error="No transcription method available",
                    method="none"
                )

        return TranscribeResponse(
            transcript=transcript,
            phone=req.phone,
            duration_seconds=round(time.time() - start_time, 1),
            method=method
        )

    except Exception as e:
        return TranscribeResponse(
            transcript="",
            phone=req.phone,
            duration_seconds=round(time.time() - start_time, 1),
            error=str(e),
            method="error"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
