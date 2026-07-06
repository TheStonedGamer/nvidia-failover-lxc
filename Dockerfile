# NVIDIA Failover Proxy — OpenAI-compatible multi-provider failover gateway.
FROM python:3.12-slim

# Faster, cleaner Python in containers.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY nvidia_failover_proxy.py .
COPY src/ ./src/

# Bind to all interfaces and keep the SQLite store on a mountable volume so
# provider config + learned stats survive container restarts/upgrades.
ENV PROXY_HOST=0.0.0.0 \
    PROXY_PORT=5002 \
    PROXY_DB_FILE=/data/proxy.db
VOLUME ["/data"]
EXPOSE 5002

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5002/health', timeout=4).status==200 else 1)"

CMD ["python", "nvidia_failover_proxy.py"]
