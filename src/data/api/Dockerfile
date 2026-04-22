FROM python:3.12-slim

WORKDIR /app

# Copy repository root so src/ layout is available in the image
COPY . /app
RUN pip install --no-cache-dir -r src/data/api/requirements.txt

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos "" appuser
USER appuser

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "data.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
