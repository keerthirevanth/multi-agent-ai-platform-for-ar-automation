FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the base-case ledger exists inside the image.
RUN PYTHONPATH=src python -m ar_platform.data.generator

ENV PYTHONPATH=/app/src
EXPOSE 8501

# Launch the executive dashboard.
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
