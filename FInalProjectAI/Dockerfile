FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ml_report_agents/ ./agents/ml_report_agents/
COPY data/ ./agents/data/
RUN useradd -m myuser && chown -R myuser /app
USER myuser
ENV PATH="/home/myuser/.local/bin:$PATH"
CMD ["sh", "-c", "adk web --host 0.0.0.0 --port ${PORT:-8080} agents"]
