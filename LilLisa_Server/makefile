SHELL = /bin/bash

.EXPORT_ALL_VARIABLES:

.PHONY: _include-env

_include-env:
    include ./env/lillisa_server.env

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
	docker build -f ./build/dockerfile_local -t lil-lisa-server-local:2.1.3 .

_build-cloud:
	docker build -f ./build/dockerfile_cloud -t lil-lisa-server:2.1.3 .

_build-cloud-lambda:
	docker build -f ./build/dockerfile_cloud_lambda -t lil-lisa-server-lambda:2.1.3 .

build-local: _lint _build-local

build-cloud: _lint _build-cloud

update-env: environment.yml
	conda env update --file environment.yml --prune

run-local:
	docker run -d -p 8000:8000 --name=lil-lisa-server lil-lisa-server-local:2.1.3

run-cloud:
	docker run -d -p 8000:8000 --name=lil-lisa-server lil-lisa-server:2.1.3

# has to be done in terminal
push-image-to-aws:
	AWS_ACCOUNT=$(aws sts get-caller-identity --query 'Account' --output text)
	AWS_REGION=$(aws ec2 describe-availability-zones --query 'AvailabilityZones[0].RegionName' --output text)
	aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com
	IMAGE='lil-lisa-server'
	TAG='2.1.3'
	docker tag ${IMAGE}:${TAG} ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE}:${TAG}
	docker push ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE}:${TAG}
