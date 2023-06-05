# ING 2 Project - Production

This is the production repository for the course Software Engineering. This API can receive a PDF of an identity document over a POST request and return the ID class (ID card, driver's license or passport) in the response.

## Table of Contents

- [1. Tech Stack](#1-tech-stack)
  - [Production](#production)
  - [Development](#development)
- [2. API Reference](#2-api-reference)
  - [POST Identity document](#post-identity-document)
- [3. Adding the trained model](#3-adding-the-trained-model)
- [4. Run in Docker Compose - Development Mode](#4-run-in-docker-compose---development-mode)
- [5. Run in Docker Compose - Production Mode](#5-run-in-docker-compose---production-mode)
- [6. Project CI](#6-project-ci)
  - [Unit Tests Report](#unit-tests-report)

## 1. Tech Stack

### Production

- Python 3.9
  - FastAPI
  - pdf2image
  - Tensorflow

### Development

- Python 3.9
  - Pytest

## 2. API Reference

This project is used using the an API.

### POST Identity document

```http
  POST /document/
```

| Parameter  | Type   | Description                              |
| :--------- | :----- | :--------------------------------------- |
| `document` | `file` | **Required**. PDF Identity document to be classified |

## 3. Adding the trained model

Inside the `ing-nn-trainer` application, there is a directory called `model_export` which contains the trained model. This model needs to be copied into a directory called `models` in the root directory of this project.

First create the `models` directory:

```terminal
mkdir models
```

Then copy all of the contents of the `model_export` directory into the `models` directory in this project.

## 4. Run in Docker Compose - Development Mode

**ING Employees please use production mode**

This mode runs the FastAPI app with hot reload enabled so any changes made to the code will be reflected in the docker container.

Create the env file from `.env.example`:

```terminal
cp .\.env.example .\.env
```

Build the docker compose and then run it:

```terminal
docker-compose pull
docker-compose up -d
```

View the logs in follow mode:

```terminal
docker-compose logs -f
```

Run tests:

```terminal
docker-compose exec app pytest
```

## 5. Run in Docker Compose - Production Mode

This mode runs the fast api app with hot reload disabled.

Create the env file from `.env.example`:

```terminal
cp .\.env.example .\.env
```

Build the docker compose and then run it:

```terminal
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
```

Or do that in one command:

```terminal
docker-compose -f docker-compose.prod.yml up -d --build
```

View the logs in follow mode:

```terminal
docker-compose -f docker-compose.prod.yml logs -f
```

## 6. Project CI

This project uses GitLab CI to ensure code quality.

The pipeline builds the docker image, and runs code style checks and then runs tests.

### Unit Tests Report

To generate a unit tests report for sonarqube, run the following command:

```terminal
docker-compose exec app /bin/bash generate_unit_tests_report.sh
```
