.PHONY: test mypy pylint

test:
	python -m unittest

mypy:
	mypy gym_gridverse/ tests/

pylint:
	pylint gym_gridverse/ tests/
