# Identity Document Classification, Part 3/3: API

This is the continued repository for the identity document (ID) classification API. It was created by Group ING 2 as a part of the 2022-2023 Software Engineering course at the University of Groningen and was done in collaboration with ING. The API can receive a PDF of an ID over a POST request and return the ID class (ID card, driving license or passport) in the response.

## Table of Contents

- [Identity Document Classification, Part 3/3: API](#identity-document-classification-part-33-api)
  - [Table of Contents](#table-of-contents)
  - [1. Tech Stack](#1-tech-stack)
    - [Production](#production)
    - [Development](#development)
  - [2. API Specification](#2-api-specification)
  - [3. Adding trained models](#3-adding-trained-models)
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

## 2. API Specification

They can be found [here](spec.yml).

## 3. Adding trained models

Inside the root directory of the `ing-nn-trainer` application, there is a directory called `model_export` which contains the trained models. These models need to be copied into a directory called `models` in the root directory of this project. Please do as follows:

First create the `models` directory:

```terminal
mkdir models
```

Then, copy all of the contents (not the directory itself!) from the `model_export` directory into the `models` directory.

## 4. Run in Docker Compose - Development Mode

**ING Employees, please use Production Mode!**

This mode runs the FastAPI app with hot reload enabled so any changes made to the code will be reflected in the docker container.

Create the env file from `.env.example`:

```terminal
cp .\.env.example .\.env
```

Build the docker compose and then run it:

```terminal
docker-compose build
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

Pull the docker-compose image and then run it:

```terminal
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
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
