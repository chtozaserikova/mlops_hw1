FROM python:3.10
ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_DEFAULT_TIMEOUT=10000 \
    PIP_INDEX_URL=https://pypi.org/simple
WORKDIR /app
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    git \
    build-essential \
    # libgomp1 \  
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "poetry==1.7.1"
COPY pyproject.toml poetry.lock ./
RUN poetry config installer.max-workers 1
RUN poetry install --no-interaction --no-ansi --no-root
COPY . .
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. app.proto
CMD ["bash"]