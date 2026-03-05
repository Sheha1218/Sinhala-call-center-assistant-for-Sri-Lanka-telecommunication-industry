# Sinhala Call Center Assistant - Sri Lanka Telecommunication (lightweight)
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install PyTorch CPU-only first (~200MB vs ~2.5GB for full CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install (torch excluded - installed above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
