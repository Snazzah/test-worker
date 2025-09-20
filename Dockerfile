FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install Python and pip
RUN apt-get update -y && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

RUN ldconfig /usr/local/cuda-12.1/compat/

# Install uv and use it to install Python dependencies (faster, reproducible)
# Copy the lockfile / requirements first so Docker layer caching helps when code changes.
COPY uv.lock requirements.txt pyproject.toml ./

# Install uv with pip and then use uv to sync the requirements file.
# Using python3 -m pip ensures we install uv for the system python in this image.
RUN python3 -m pip install --no-cache-dir uv && uv venv && uv sync

# Copy application code
COPY app.py .

# Start the handler
CMD ["python3", "app.py"]
