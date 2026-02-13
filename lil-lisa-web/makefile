SHELL = /bin/bash

APP_NAMESPACE=lil-lisa-web

AWS_ACCOUNT ?= $(shell aws sts get-caller-identity --query 'Account' --output text 2>/dev/null)
AWS_REGION ?= us-west-2

IMAGE=lil-lisa-web
TAG=2.4.1

.EXPORT_ALL_VARIABLES:

.PHONY: _include-env-local, _include-env-cloud

_include-env-local:
    include ./lil-lisa-web.env

_include-env-cloud:
    include ./build/lil-lisa-web.env

# Create uv venv under lil-lisa-web (replaces conda env)
setup-env:
	uv venv .venv
	uv sync --extra dev

# Use venv tools (run 'make setup-env' first)
VENV_BIN = .venv/bin

_lint:
	$(VENV_BIN)/isort .
	$(VENV_BIN)/black .
	$(VENV_BIN)/flake8 . --ignore E501,E122,W503,E402
	$(VENV_BIN)/pylint --recursive=y .
	$(VENV_BIN)/mypy --install-types --non-interactive .
	$(VENV_BIN)/mypy .
	$(VENV_BIN)/bandit -c pyproject.toml -r .

_build-local:
	docker build -f ./build/Dockerfile_local -t ${IMAGE}-local:${TAG} .

_build-cloud:
	docker build -f ./build/Dockerfile_cloud -t ${IMAGE}:${TAG} .

build-local: _lint _build-local

build-cloud: _lint _build-cloud

update-env:
	uv sync --extra dev

run-local:
	docker run -d --name=${IMAGE} ${IMAGE}-local:${TAG}

run-cloud:
	docker run -d --name=${IMAGE} ${IMAGE}:${TAG}

# has to be done in terminal
push-image-to-aws:
	aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com
	docker tag ${IMAGE}:${TAG} ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE}:${TAG}
	docker push ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE}:${TAG}
