FROM python:3.12-slim

WORKDIR /app

# Copy repository root; evaluation is imported from src/ on PYTHONPATH.
COPY . /app
RUN pip install --no-cache-dir -r src/evaluation/requirements.txt

RUN adduser --disabled-password --gecos "" appuser
USER appuser

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "evaluation.run_judge"]
CMD ["--help"]
