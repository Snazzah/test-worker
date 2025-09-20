import os
import sys
import base64
import time
import json

try:
    import requests
except Exception:
    print("Missing dependency: requests. Install with: pip install requests")
    sys.exit(1)


def gather_audio_base64(audios_dir: str):
    files = sorted(
        [
            f
            for f in os.listdir(audios_dir)
            if os.path.isfile(os.path.join(audios_dir, f))
        ]
    )
    buffers = []
    filenames = []
    for fname in files:
        path = os.path.join(audios_dir, fname)
        try:
            with open(path, "rb") as fh:
                data = fh.read()
            b64 = base64.b64encode(data).decode("ascii")
            buffers.append(b64)
            filenames.append(fname)
        except Exception as e:
            print(f"Failed to read/encode {path}: {e}")
    return filenames, buffers


def main():
    base_url = os.environ.get("BASE_URL", "http://localhost:5000")
    endpoint = f"{base_url.rstrip('/')}/transcribe"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    audios_dir = os.path.join(script_dir, "audios")

    if not os.path.isdir(audios_dir):
        print(f"audios directory not found at: {audios_dir}")
        sys.exit(1)

    filenames, buffers = gather_audio_base64(audios_dir)
    if not buffers:
        print("No audio files found to send.")
        return

    payload = {"audio_buffers": buffers}

    print(f"Sending {len(buffers)} audio file(s) to {endpoint} ...")
    start = time.perf_counter()
    try:
        resp = requests.post(endpoint, json=payload, timeout=600)
    except Exception as e:
        print(f"Request failed: {e}")
        sys.exit(1)
    elapsed = time.perf_counter() - start

    print(f"HTTP {resp.status_code} — total elapsed: {elapsed:.3f}s")

    if resp.status_code != 200:
        print("Non-200 response body:")
        print(resp.text)
        sys.exit(1)

    try:
        data = resp.json()
    except Exception as e:
        print(f"Failed to decode JSON response: {e}")
        print(resp.text)
        sys.exit(1)

    results = data.get("results") or []
    if not results:
        print("No 'results' in response. Full response:")
        print(json.dumps(data, indent=2))
        return

    # Print transcription for each filename in order
    for idx, res in enumerate(results):
        fname = filenames[idx] if idx < len(filenames) else f"audio_{idx}"
        text = res.get("text", "")
        print("--------------------------------------------------")
        print(f"File: {fname}")
        print(f"Elapsed (total): {elapsed:.3f}s")
        print("Transcription:")
        print(text)
        print("--------------------------------------------------\n")

    print("Done.")


if __name__ == "__main__":
    main()
