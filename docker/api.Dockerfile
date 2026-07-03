FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# bitsandbytes usually needed if model is 4-bit, but peft fallback might run in 16-bit
RUN pip install --no-cache-dir bitsandbytes accelerate

# We copy the entire project structure so relative paths work
COPY api/ /app/api/
# Checkpoints would ideally be mounted as volume, but we create the directory here
RUN mkdir -p /app/checkpoints

WORKDIR /app/api
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
