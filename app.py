import base64
from contextlib import asynccontextmanager
import tempfile
import os
import threading
import asyncio
import time
from typing import List, Optional
from fastapi import FastAPI, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from faster_whisper.transcribe import TranscriptionInfo, Segment
from pydantic import BaseModel
from urllib.parse import urlparse
import urllib.request
from dotenv import load_dotenv

load_dotenv()

model: Optional[WhisperModel] = None
CONCURRENCY = int(os.getenv("TRANSCRIBE_CONCURRENCY", os.getenv("CONCURRENCY", "2")))
transcribe_semaphore = asyncio.Semaphore(CONCURRENCY)

async def load_model():
    global model
    model_name = os.getenv("MODEL_NAME", "turbo")
    device_type = os.getenv("DEVICE_TYPE", "cuda")  # "cpu", "cuda", "auto"
    compute_type = os.getenv("COMPUTE_TYPE", "float16")  # https://opennmt.net/CTranslate2/quantization.html
    print(f"Loading model: {model_name}...")
    try:
        model = WhisperModel(
            model_name,
            device=device_type,
            compute_type=compute_type,
        )
        print(f"Model {model_name} loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise ValueError(f"Failed to load model {model_name}: {e}") from e

def base64_to_tempfile(base64_file: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))

    return temp_file.name

def buffer_or_url_to_tempfile(buf: str) -> str:
    """
    Accepts either a base64-encoded audio string or an HTTP/HTTPS URL to an audio file.
    Downloads or decodes into a temporary file and returns its path.
    """
    try:
        parsed = urlparse(buf)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            suffix = os.path.splitext(parsed.path)[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf, urllib.request.urlopen(buf, timeout=30) as resp:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    tf.write(chunk)
            return tf.name
    except Exception:
        # Fall back to base64 handling if URL parsing or download fails
        pass

    return base64_to_tempfile(buf)

def _run_transcribe_sync(audio_file: str) -> tuple[List[Segment], TranscriptionInfo]:
    global model
    if model is None:
        raise RuntimeError("Model not loaded")
    segments, info = model.transcribe(
        audio_file,
        task="transcribe",
        # log_progress=True,
        beam_size=5,
        # best_of=5,
        # patience=1,
        # length_penalty=None,
        # temperature=tuple(np.arange(0, 1.0 + 1e-6, 0.2)),
        # compression_ratio_threshold=2.4,
        # log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        # condition_on_previous_text=True,
        suppress_blank=True,
        # suppress_tokens=[-1],
        without_timestamps=True,
        # max_initial_timestamp=1.0,
        word_timestamps=False,
        vad_filter=True,
    )
    return list(segments), info

async def transcribe_clip(b64_or_url: str) -> "TranscriptionClipResponse":
    path = await asyncio.to_thread(buffer_or_url_to_tempfile, b64_or_url)
    try:
        start_time = time.time()
        async with transcribe_semaphore:
            segments, info = await asyncio.to_thread(_run_transcribe_sync, path)
        transcription_time = time.time() - start_time
        text = " ".join([segment.text.lstrip() for segment in segments])
        output_tokens = sum(len(segment.tokens) for segment in segments)
        print(f"Transcribed {info.duration}s clip, took {transcription_time}s, {output_tokens} tokens")
        return TranscriptionClipResponse(text=text, duration=info.duration, output_tokens=output_tokens, transcription_time=transcription_time)
    finally:
        try:
            if path and os.path.exists(path):
                await asyncio.to_thread(os.remove, path)
        except Exception:
            pass

def _start_loader_thread():
    try:
        asyncio.run(load_model())
    except Exception as e:
        print(f"Background model loader failed: {e}")

threading.Thread(target=_start_loader_thread, daemon=True).start()

@asynccontextmanager
async def lifespan(_: FastAPI):
    yield

    global model
    if model:
        model = None

app = FastAPI(title="Whisper Load Balancer", lifespan=lifespan)

class TranscriptionRequest(BaseModel):
    audio_buffers: list[str]

class TranscriptionClipResponse(BaseModel):
    text: str
    duration: float
    output_tokens: int
    transcription_time: float

async def verify_api_key(authorization: Optional[str] = Header(None)):
    """
    If API_KEY env var is set, require the same key in the Authorization header.
    Accepts the raw key or 'Bearer <key>'.
    """
    api_key = os.getenv("API_KEY")
    # If no API_KEY is configured, skip verification
    if not api_key:
        return

    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    # Accept the raw key or "Bearer <key>"
    if authorization == api_key:
        return
    if authorization.startswith("Bearer ") and authorization.split(" ", 1)[1] == api_key:
        return

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

request_count = 0

@app.get("/ping", dependencies=[Depends(verify_api_key)])
async def health_check():
    if model is None:
        return JSONResponse(
            content={"status": "initializing"},
            status_code=204
        )

    return {"status": "healthy"}

@app.post("/transcribe", dependencies=[Depends(verify_api_key)])
async def transcribe(request: TranscriptionRequest):
    global request_count, model
    request_count += 1

    if model is None:
        return JSONResponse(
            content={
                "error": "ServiceUnavailable",
                "message": "Model not ready"
            },
            status_code=503
        )

    start_time = time.time()
    tasks = [transcribe_clip(b64_or_url) for b64_or_url in request.audio_buffers]
    results_models = await asyncio.gather(*tasks)
    results = [r.model_dump() for r in results_models]
    request_time = time.time() - start_time
    print(f"Request with {len(results)} clips took {request_time}s")
    return {"results": results}

# A simple endpoint to show request stats
@app.get("/stats", dependencies=[Depends(verify_api_key)])
async def stats():
    return {"total_requests": request_count}

# Run the app when the script is executed
if __name__ == "__main__":
    import uvicorn

    # When you deploy the endpoint, make sure to expose port 5000
    # And add it as an environment variable in the Runpod console
    port = int(os.getenv("PORT", "5000"))

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)
