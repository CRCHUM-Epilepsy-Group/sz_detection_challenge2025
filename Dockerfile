FROM python:3.11
COPY --from=ghcr.io/astral-sh/uv:0.5.9 /uv /uvx /bin/

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
RUN uv pip install -e .

# Install epileptology package
RUN uv pip install /app/epileptology

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Explicitly mention volumes
VOLUME ["/data"]
VOLUME ["/output"]

# Explicitely define environment variables
ENV INPUT=""
ENV OUTPUT=""
ENV IN_DOCKER=1

# Run the pipeline (to change later!)
CMD ["uv", "run", "test_main.py"]
