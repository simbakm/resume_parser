# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY resume_microservice.py .
COPY download_model.py .

# Copy the resume_parser directory (including models)
# COPY resume_parser/ ./resume_parser/

# Create necessary directories
RUN mkdir -p /tmp/uploads

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=resume_microservice.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Run the application
CMD ["python", "resume_microservice.py"]
