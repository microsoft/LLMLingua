.PHONY: install style test

PYTHON := python
CHECK_DIRS := llmlingua tests

install:
	@${PYTHON} setup.py bdist_wheel
	@${PYTHON} -m pip install dist/sdtools*

style:
	black $(CHECK_DIRS)
	isort -rc $(CHECK_DIRS)
	flake8 $(CHECK_DIRS)

test:
	@${PYTHON} -m pytest -n auto --dist=loadfile -s -v ./tests/
