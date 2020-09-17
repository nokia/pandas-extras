.PHONY: lint
lint:
	@python -m pylint pandas_extras


.PHONY: test
test:
	@python -m unittest discover tests


.PHONY: coverage
coverage:
	@python -m coverage run --branch --module unittest discover tests
	@python -m coverage report --show-missing --include "**/pandas_extras/**"


.PHONY: docs
docs:
	@python -m sphinx -b html docs docs/_build


.PHONY: publish
publish:
	@python setup.py sdist bdist_wheel
	@twine upload dist/*
