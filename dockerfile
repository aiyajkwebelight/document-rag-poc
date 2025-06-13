# Builder Stage
FROM python:3.12.3-slim AS builder

# Copy project files
COPY ./poetry.lock ./pyproject.toml /code/
COPY ./src /code/src
COPY .env /code/.env

# Set working directory
WORKDIR /code

# Install Python dependencies using poetry
# hadolint ignore=DL3013
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-root && \
    pip uninstall -y poetry

# Runner Stage
FROM python:3.12-slim AS runner

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "/code/src:${PYTHONPATH}"

# Copy files from the builder stage
COPY --from=builder /code /code
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install curl for healthcheck
# hadolint ignore=DL3013
# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Healthcheck
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 CMD curl -f http://localhost:8501/healthcheck || exit 1

# hadolint ignore=DL3059
RUN useradd -m -s /bin/bash app

# hadolint ignore=DL3059
RUN usermod -aG root app

USER app

# Set working directory
WORKDIR /code

# Entrypoint for Streamlit
ENTRYPOINT ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]