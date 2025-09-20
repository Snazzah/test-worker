FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.1/compat/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["python3", "app.py"]
