# ING 2 Project - Proof of Concept

This is the POC for the course Software Engineering. This application receives pdf documents over a POST request and returns the document type (ID card, driver's license or passport).

## Table of Contents

* [1. Tech Stack](#1-tech-stack)
* [2. API Reference](#2-api-reference)
* [3. Run in Docker Compose - Development Mode](#3-run-in-docker-compose---development-mode)
* [4. Run in Docker Compose - Production Mode](#4-run-in-docker-compose---production-mode)
* [5. Project CI](#5-project-ci)

## 1. Tech Stack

- Python 3.9
- FastAPI
- Tensorflow

### Development

- Pytest (Unit test framework)
- Black (Code formatter)
- isort (Dependency manager)

## 2. API Reference

This project is used using the an API.

### POST Identity document

```http
  POST /document/
```

| Parameter  | Type   | Description                              |
| :--------- | :----- | :--------------------------------------- |
| `document` | `file` | **Required**. Document to be classified |

## 3. Run in Docker Compose - Development Mode

This mode runs the fast api app with hot reload enabled, so any changes made to the code will be reflected in the docker container and the code will hot reload.

Create the env file from `.env.example`:

```terminal
cp .\.env.example .\.env
```

Build the docker compose and then run it:

```terminal
docker-compose build
docker-compose up -d
```

Or do that in one command:

```terminal
docker-compose up -d --build
```

View the logs in follow mode:

```terminal
docker-compose logs -f
```

Run tests:

```terminal
docker-compose exec app pytest
```

## 4. Run in Docker Compose - Production Mode

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

## 5. Project CI

This project uses GitLab CI to ensure code quality.

The pipeline builds the docker image, and runs code style checks and then runs tests.

### Linting

This project uses `black` and `isort` for python linting. Before committing to the git repository, please run the following to ensure compliance:

```terminal
docker-compose exec app /bin/bash run_linters.sh
```

Or use docker compose run:

```terminal
docker-compose run app /bin/bash run_linters.sh
```
