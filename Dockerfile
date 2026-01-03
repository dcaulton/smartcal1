FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Fix tzdata interactive prompt – set timezone early
ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Python 3.11.9 exactly
RUN apt-get update && apt-get install -y wget software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm -rf /var/lib/apt/lists/* get-pip.py

RUN apt-get update && apt-get install -y sqlite3 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3.11 -m venv /app/venv && \
    . /app/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    pip install -r requirements.txt

COPY src/ .
ENV PATH="/app/venv/bin:$PATH"
ENTRYPOINT ["python3.11", "app.py"]
