SHELL = /bin/bash

APP_NAMESPACE=lil-lisa

AWS_ACCOUNT = $(shell aws sts get-caller-identity --query 'Account' --output text)
AWS_REGION = $(shell aws ec2 describe-availability-zones --query 'AvailabilityZones[0].RegionName' --output text)

IMAGE=lil-lisa
TAG=2.1.7

.EXPORT_ALL_VARIABLES:

.PHONY: _include-env

_include-env:
    include ./app_envfiles/${IMAGE}.env

_wait30s:
	@echo "lets wait 30 sec..."
	sleep 30s

condaenv:
	conda env create -f environment.yml
	conda activate ${APP_NAMESPACE}

_lint:
	py3clean .
	isort .
	black .
	flake8 . --ignore E501,E122,W503,E402
	pylint --recursive=y .
	mypy --install-types --non-interactive .
	mypy .
	bandit -c pyproject.toml -r .

_build-local:
	docker build -f ./build/dockerfile_local -t ${IMAGE}-local:${TAG} .

_build-cloud:
	docker build -f ./build/dockerfile_cloud -t ${IMAGE}:${TAG} .

build-local: _lint _build-local

build-cloud: _lint _build-cloud

update-env: environment.yml
	conda env update --file environment.yml --prune

run-local:
	docker run -d --name=${IMAGE} ${IMAGE}-local:${TAG}

run-cloud:
	docker run -d --name=${IMAGE} ${IMAGE}:${TAG}

# has to be done in terminal
push-image-to-aws:
	aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com
	docker tag ${IMAGE}:${TAG} ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE}:${TAG}
	docker push ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE}:${TAG}
