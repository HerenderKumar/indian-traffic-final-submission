# Lock to the stable 'bookworm' release so packages never randomly disappear
FROM python:3.10-slim-bookworm

# Prevent Python from buffering stdout/stderr (vital for logs)
ENV PYTHONUNBUFFERED=1

# Install system dependencies and SUMO Traffic Simulator
RUN apt-get update && apt-get install -y \
    sumo \
    sumo-tools \
    && rm -rf /var/lib/apt/lists/*

# CRITICAL: Set the SUMO_HOME variable so env/ scripts can find TraCI
ENV SUMO_HOME=/usr/share/sumo

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install the CPU version of PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy your entire project structure into the container
COPY . .

# Expose the exact port defined in your openenv.yaml
EXPOSE 7860

# Run the FastAPI server continuously
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]