FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch keeps the image small; installed first for layer caching
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[dev]"

CMD ["bash"]
