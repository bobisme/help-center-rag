set shell := ["bash", "-c"]

fmt:
  black epic_rag/**/*.py

lint:
  flake8 --max-complexity 10 --max-line-length 88 epic_rag/
