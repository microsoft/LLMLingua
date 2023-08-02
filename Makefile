.PHONY: install style_check_on_modified style

export PYTHONPATH = src

PYTHON := python3
CHECK_DIRS := llmlingua

install:
	@${PYTHON} setup.py bdist_wheel
	@${PYTHON} -m pip install dist/sdtools*

style:
	black $(CHECK_DIRS)
	isort -rc $(CHECK_DIRS)
	flake8 $(CHECK_DIRS)