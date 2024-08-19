SHELL = /bin/bash

.EXPORT_ALL_VARIABLES:

.PHONY: _include-env

_include-env:
    include lil-lisa-web.env

condaenv:
	conda env create -f environment.yml
	# conda init
	# conda activate ${APP_NAMESPACE}

_lint:
	py3clean .
	isort .
	black .
	flake8 . --ignore E501,E122,W503,E402
	pylint --recursive=y .
	mypy --install-types --non-interactive .
	mypy .
	bandit -c pyproject.toml -r .

update-env: environment.yml	## start with miniconda installed or docker image continuumio/miniconda3:22.11.1
	conda env update --file environment.yml --prune