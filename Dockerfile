# Use an official Python runtime as a parent image
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Set the working directory in the container
WORKDIR /app

# # Install system dependencies (curl is needed to download uv)
# RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# # Install uv (The tool you are using)
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# ENV PATH="/root/.cargo/bin:$PATH"

# Copy the project files
COPY pyproject.toml .
COPY uv.lock .
COPY README.md .

# Copy your source code folders
COPY api/ ./api/
COPY feature_engineering/ ./feature_engineering/
COPY spam/ ./spam/
COPY models/ ./models/
COPY training/ ./training/
COPY sql/ ./sql/

# Install dependencies (System-wide, no venv needed inside Docker)
# We install the 'api' optional group we defined earlier
RUN uv pip install --system ".[api]"
RUN uv sync --locked

# Expose the port the app runs on
EXPOSE 8080

# Run the application
# Cloud Run expects the app to listen on the PORT environment variable (default 8080)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]