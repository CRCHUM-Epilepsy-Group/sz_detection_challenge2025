FROM python:3.11 AS builder
COPY --from=ghcr.io/astral-sh/uv:0.5.29 /uv /uvx /bin/

LABEL org.opencontainers.image.source=https://github.com/CRCHUM-Epilepsy-Group/sz_detection_challenge2025

# Install cmake for epileptology further down
RUN apt-get -y update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    cmake

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Install the szdetect package
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install .

# Install epileptology package
RUN --mount=type=bind,source=epileptology,target=/pkg/epileptology \
    uv pip install /pkg/epileptology

FROM python:3.11

ARG TAU
ARG THRESHOLD

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.5.29 /uv /uvx /bin/
COPY --from=builder /app/.venv/ /app/.venv/
ADD . /app
RUN rm -rf epileptology

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Explicitly mention volumes
VOLUME ["/data"]
VOLUME ["/output"]

# Explicitely define environment variables
ENV INPUT=""
ENV OUTPUT=""

ENV IN_DOCKER=1
ENV TAU=${TAU}
ENV THRESHOLD=${THRESHOLD}

CMD ["uv", "run", "main.py"]
