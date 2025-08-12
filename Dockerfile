# Dockerfile
FROM python:3.11-slim

# Set environment variables for better pip behavior
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libpq-dev \
        sqlite3 \
        git \
        gcc \
        g++ \
        python3-dev \
        pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt /app/

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/static /app/media /app/db

# Set permissions
RUN chmod +x manage.py entrypoint.sh

# Expose port
EXPOSE 8000

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

# Command to run the application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]