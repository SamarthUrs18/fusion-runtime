# fusion-runtime Makefile

.PHONY: help install install-dev download-models run run-cpu test lint format clean deploy-modal setup-modal

help:
	@echo "fusion-runtime - Low-latency voice AI inference runtime"
	@echo ""
	@echo "Commands:"
	@echo "  install         Install production dependencies"
	@echo "  install-dev     Install with dev dependencies"
	@echo "  download-models Download all models to /models"
	@echo "  run             Run with Docker Compose (GPU)"
	@echo "  run-cpu         Run CPU-only with Docker Compose"
	@echo "  test            Run tests"
	@echo "  lint            Run ruff + mypy"
	@echo "  format          Format with ruff"
	@echo "  clean           Clean build artifacts"
	@echo "  deploy-modal    Deploy to Modal"
	@echo "  setup-modal     Download models to Modal volume"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

download-models:
	python scripts/download_models.py --all

run:
	cd docker && docker-compose up -d

run-cpu:
	cd docker && docker-compose up fusion-runtime-cpu

run-native:
	python3 -m uvicorn fusion_runtime.server:app --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

test-integration:
	pytest tests/ -v -m integration

lint:
	ruff check fusion_runtime/
	mypy fusion_runtime/

format:
	ruff check --fix fusion_runtime/
	ruff format fusion_runtime/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info

deploy-modal:
	modal deploy modal_deploy.py

setup-modal:
	modal run modal_deploy.py::setup_models

# Development shortcuts
dev: install-dev download-models run-cpu

logs:
	cd docker && docker-compose logs -f fusion-runtime

shell:
	cd docker && docker-compose exec fusion-runtime bash

# Model shortcuts
download-whisper:
	python scripts/download_models.py --whisper

download-llm:
	python scripts/download_models.py --llm

download-kokoro:
	python scripts/download_models.py --kokoro

download-firered:
	python scripts/download_models.py --firered-asr --firered-tts --firered-eot