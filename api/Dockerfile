FROM python:3.9-slim

WORKDIR /app

# set environment variables - Stops generation of Pycache
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install -y poppler-utils && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt

# COPY generate_unit_tests_report.sh /generate_unit_tests_report.sh

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${port:-8000}", "--app-dir", "./app/api/src/", "--reload" ]
