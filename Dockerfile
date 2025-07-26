FROM python:3.10-slim

# Copy all requirements files
COPY shared.txt shared.txt
COPY api/requirements.txt api/requirements.txt
COPY model/requirements.txt model/requirements.txt

# Install dependencies
RUN pip install -r shared.txt -r api/requirements.txt -r model/requirements.txt

# Make sure data directory is copied
COPY data/ /data/

# Set Python path to include the app root
ENV PYTHONPATH=/

# Copy application code
COPY api api
COPY model model

# Cloud Run provides PORT environment variable
EXPOSE $PORT
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
