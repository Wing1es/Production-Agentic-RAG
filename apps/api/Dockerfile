FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

# Set working directory inside the container
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy application source code
COPY src/ ./src/

# Expose FastAPI's default port
EXPOSE 8000

# Run the FastAPI application on container start
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]