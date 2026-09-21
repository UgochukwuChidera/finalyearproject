FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

ENV FLASK_DEBUG=0
EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "5", "-b", "0.0.0.0:8000", "web:create_app()"]
