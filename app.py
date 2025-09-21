import base64
import tempfile
from contextlib import asynccontextmanager
import os
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel

model: Optional[WhisperModel] = None

async def load_model():
    global model
    model_name = os.getenv("MODEL_NAME", "turbo")
    device_type = os.getenv("DEVICE_TYPE", "cuda") # "cpu", "cuda", "auto"
    compute_type = os.getenv("COMPUTE_TYPE", "float16") # https://opennmt.net/CTranslate2/quantization.html
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
    

@asynccontextmanager
async def lifespan(_: FastAPI):
    await load_model()
    yield

    global model
    if model:
        model = None

app = FastAPI(title="Whisper Load Balancer", lifespan=lifespan)

class TranscriptionRequest(BaseModel):
    audio_buffers: list[str]

request_count = 0

@app.get("/ping")
async def health_check():
    if model is None:
        return JSONResponse(
            content={"status": "initializing"},
            status_code=204
        )

    return {"status": "healthy"}

@app.post("/transcribe")
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

    audio_files = [base64_to_tempfile(b64) for b64 in request.audio_buffers]
    results = []
    for audio_file in audio_files:
        segments, info = list(
            model.transcribe(
                audio_file,
                task="transcribe",
                # log_progress=True,
                # beam_size=5,
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
                without_timestamps=False,
                # max_initial_timestamp=1.0,
                word_timestamps=False,
                vad_filter=True,
            )
        )
        results.append({
            "text": " ".join([segment.text.lstrip() for segment in segments])
        })
    
    for path in audio_files:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    return {"results": results}

# A simple endpoint to show request stats
@app.get("/stats")
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