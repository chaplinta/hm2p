# hm2p Makefile — common development targets

.PHONY: help install test lint fmt dry-run docker-build ecr-push download-test-session

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

install:  ## Install package + dev dependencies
	pip install -e ".[dev]"

install-all:  ## Install package + dev + workflow dependencies
	pip install -e ".[dev,workflow]"

test:  ## Run pytest with coverage
	python -m pytest tests/ -q --tb=short

test-cov:  ## Run pytest with full coverage report
	python -m pytest tests/ --cov=hm2p --cov-report=term-missing --cov-fail-under=90

lint:  ## Run ruff linter
	ruff check src/ tests/

fmt:  ## Auto-format with ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:  ## Run mypy type checker
	mypy src/hm2p/

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

dry-run:  ## Snakemake dry-run (resolve DAG, no execution)
	cd workflow && snakemake -n --profile profiles/local

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build:  ## Build CPU and GPU Docker images locally
	docker build -f docker/cpu.Dockerfile -t hm2p-cpu .
	docker build -f docker/gpu.Dockerfile -t hm2p-gpu .

ecr-push:  ## Build + push Docker images to ECR
	./scripts/ecr_push.sh

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

download-test-session:  ## Download one test session from S3
	python scripts/download_from_s3.py \
		--session 20221115_13_27_42_1118213 \
		--data-root data \
		--yes
