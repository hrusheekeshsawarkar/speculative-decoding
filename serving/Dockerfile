# Python packages (including the matching PyTorch CUDA runtime) are installed on first
# container start into the PVC (/cache/pylibs) and reused on restarts.
FROM python:3.12-slim

WORKDIR /app
COPY serving/requirements.txt serving/entrypoint.sh serving/server.py ./
COPY src/speculative_decoding ./src/speculative_decoding
RUN chmod +x entrypoint.sh

ENV HF_HOME=/cache/hf \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["/app/entrypoint.sh"]
