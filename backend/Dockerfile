# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements/  requirements/
RUN pip install --no-cache-dir -r requirements/project.txt
RUN pip install --no-cache-dir -r requirements/dev.txt
# Copy the FastAPI application
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
