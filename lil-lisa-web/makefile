SHELL = /bin/bash

APP_NAMESPACE=lil-lisa-web

AWS_ACCOUNT = $(shell aws sts get-caller-identity --query 'Account' --output text)
AWS_REGION = us-west-2#$(shell aws ec2 describe-availability-zones --query 'AvailabilityZones[0].RegionName' --output text)

IMAGE=lil-lisa-web
TAG=2.2.4

.EXPORT_ALL_VARIABLES:

.PHONY: _include-env-local, _include-env-cloud

_include-env-local:
    include ./lil-lisa-web.env

_include-env-cloud:
    include ./build/lil-lisa-web.env

condaenv:
	conda env create -f environment.yml
	conda init
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
	docker build -f ./build/Dockerfile_local -t ${IMAGE}-local:${TAG} .

_build-cloud:
	docker build -f ./build/Dockerfile_cloud -t ${IMAGE}:${TAG} .

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
