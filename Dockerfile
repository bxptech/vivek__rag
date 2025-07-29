FROM ubuntu:22.04

# System packages
RUN apt-get update && apt-get install -y \
    wget build-essential curl zip unzip git \
    libopenblas-dev libomp-dev \
    libgl1 python3-pip python3-venv python3-dev jq

# Optional: Install Python 3.13.5 manually
RUN apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.13 python3.13-dev python3.13-venv && \
    ln -s /usr/bin/python3.13 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Create venv and install deps
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Streamlit port
EXPOSE 8501

# Run app
CMD ["/app/venv/bin/streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
