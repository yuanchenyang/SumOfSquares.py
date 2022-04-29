.PHONY: build test upload install-local

build:
	python -m build

test:
	pytest

upload:
	python -m twine upload --repository pypi dist/*

install-local:
	python -m pip install -e .
