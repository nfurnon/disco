SHELL := /bin/bash

.PHONY: all install clean coverage tests doc

all: coverage doc

install:
	pip install -U setuptools pip
	pip install -e .
	pip install -r requirements.txt
	pip install pyroomacoustics  # depends on numpy, cannot put it in requirements.txt

clean:
	rm -f .coverage
	rm -rf .pytest_cache

coverage: tests
	coverage html

tests:
	coverage run -m pytest

doc:
	cd doc; make
