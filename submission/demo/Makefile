UV   ?= uv
PORT ?= 3000
HOST ?= 127.0.0.1
UV_NO_CACHE ?= 1
export UV_NO_CACHE

BE_DIR  := apps/backend
FE_DIR  := apps/frontend
PKG_DIR := packages

ifeq ($(OS),Windows_NT)
PYTHONPATH := packages;apps/backend
else
PYTHONPATH := packages:apps/backend
endif
export PYTHONPATH

.DEFAULT_GOAL := help

# ─── Help ───────────────────────────────────────────────────────
.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ─── Install ────────────────────────────────────────────────────
.PHONY: install
install: ## Install Python dependencies
	$(UV) sync

# ─── Dev ────────────────────────────────────────────────────────
.PHONY: dev dev-be
dev: ## Start backend + open browser (frontend served at /:$(PORT))
	@echo "[demo] Starting backend on http://$(HOST):$(PORT) ..."
	@$(UV) run python $(BE_DIR)/open_when_ready.py http://$(HOST):$(PORT) 90 &
	$(UV) run python -m uvicorn app:app --app-dir $(BE_DIR) --host $(HOST) --port $(PORT) --reload --reload-dir apps --reload-dir packages

dev-be: ## Start backend only (hot-reload)
	$(UV) run python -m uvicorn app:app --app-dir $(BE_DIR) --host $(HOST) --port $(PORT) --reload --reload-dir apps --reload-dir packages

# ─── Check ──────────────────────────────────────────────────────
.PHONY: check check-py check-js
check: check-py check-js ## Syntax-check Python and JS sources

check-py: ## Syntax-check Python sources
	@echo "[check] Python syntax..."
	@$(UV) run python -m compileall -q $(BE_DIR) $(PKG_DIR) && echo "  OK  Python sources"

check-js: ## Syntax-check JS sources (requires node)
	@echo "[check] JavaScript syntax..."
	@if command -v node >/dev/null 2>&1; then \
	    node --check $(FE_DIR)/app.js && echo "  OK  app.js"; \
	else \
	    echo "  SKIP  node not found"; \
	fi

# ─── Clean ──────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove __pycache__, *.pyc, and logs
	@echo "[clean] Removing caches and logs..."
	@find . -type d -name __pycache__ -not -path './.git/*' | xargs rm -rf
	@find . -name "*.pyc" -not -path './.git/*' -delete
	@rm -rf $(BE_DIR)/logs
	@echo "[clean] Done."
