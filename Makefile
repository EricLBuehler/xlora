.PHONY: quality style publish

check_dirs := src tests examples

# Runs checks on all files
quality:
	ruff $(check_dirs)
	ruff format --check $(check_dirs)

# Format source code automatically and check is there are any problems left that need manual fixing
style:
	ruff $(check_dirs) --fix
	ruff format $(check_dirs)

publish:
	python3 -m build
	twine upload dist/*