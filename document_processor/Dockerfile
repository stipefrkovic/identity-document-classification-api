FROM python:3.9

WORKDIR /app

# set environment variables - Stops generation of Pycache
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install -y poppler-utils && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

COPY ./run_linters.sh /run_linters.sh
COPY ./.flake8 /.flake8

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${port:-8000}", "--app-dir", "./app/src/", "--reload" ]