.PHONY: test mypy pylint ctags

test:
	python -m unittest

mypy:
	mypy gym_gridverse/ tests/

pylint:
	pylint gym_gridverse/ tests/

ctags:
	ctags -R .
