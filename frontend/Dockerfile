# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/project.txt
RUN pip install --no-cache-dir -r requirements/dev.txt

# Copy the Streamlit application
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Start Streamlit server
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
