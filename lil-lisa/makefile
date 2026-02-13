SHELL = /bin/bash

APP_NAMESPACE=lil-lisa

AWS_ACCOUNT ?= $(shell aws sts get-caller-identity --query 'Account' --output text 2>/dev/null)
AWS_REGION ?= us-west-2

IMAGE=lil-lisa
TAG=2.3.2

.EXPORT_ALL_VARIABLES:

.PHONY: _include-env

_include-env:
    include ./app_envfiles/${IMAGE}.env

_wait30s:
	@echo "lets wait 30 sec..."
	sleep 30s

# Create uv venv under lil-lisa (replaces conda env)
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
	docker build -f ./build/dockerfile_local -t ${IMAGE}-local:${TAG} .

_build-cloud:
	docker build -f ./build/dockerfile_cloud -t ${IMAGE}:${TAG} .

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

# has to be done in terminal
pull-image-from-aws:
	aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com
	docker pull ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE}:${TAG}
