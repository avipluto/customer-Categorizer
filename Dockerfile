# Use a slim Python base image to keep the final image size down
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
# (makes logs show up immediately, useful for debugging)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container.
# This MUST match where the app expects to find config/, templates/, etc.,
# since several config paths (schema.yaml, model.yaml, prediction_schema.yaml)
# are resolved as relative paths from the current working directory.
WORKDIR /app

# Install system-level build dependencies needed by some Python packages
# (e.g. catboost, scikit-learn compile some components from source on
# certain platforms). build-essential covers gcc/make; libgomp1 is
# needed by catboost's OpenMP-based training at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first (not the whole project) so Docker can
# cache this layer - dependencies won't be reinstalled on every rebuild
# unless requirements.txt itself changes.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project into the image
COPY . .

# Document that the app listens on port 8080 (informational - actual
# publishing happens via `docker run -p`)
EXPOSE 8080

# Run the app. Using the same entrypoint as local development
# (app.py's __main__ block, which calls uvicorn.run with host="0.0.0.0")
CMD ["python", "app.py"]
