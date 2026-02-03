FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY deploy/container_apps/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "fraud_detection.serving.container_registry.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
