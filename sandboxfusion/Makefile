HOST ?= 127.0.0.1
PORT ?= 8080
TEST_NP ?= 4
run:
	uvicorn sandbox.server.server:app --reload --host $(HOST) --port $(PORT)

run-online:
	- pkill -f "sandbox_clean.py" || true
	- bash -c 'pids=$$(lsof -ti:$(PORT) -sTCP:LISTEN 2>/dev/null || true); if [ -n "$$pids" ]; then kill -9 $$pids || true; fi'
	nohup python3 sandbox_clean.py > sandbox_clean.log 2>&1 &
	uvicorn sandbox.server.server:app --host 127.0.0.1 --port $(PORT)  --workers 32 --loop uvloop --http httptools

build-server-image:
	docker build . -f scripts/Dockerfile.server -t sandbox:server

test:
	pytest -m "not cuda and not datalake and not dp_eval and not lean" -n $(TEST_NP)

test-cuda:
	pytest -m cuda

test-minor:
	pytest -m minor

test-verilog:
	pytest -m verilog

test-verilog-pdb:
	pytest -m verilog --pdb --capture=no

test-online:
	ONLINE_TEST=1 pytest

test-case:
	pytest -s -vv -k $(CASE)

format:
	pycln --config pyproject.toml
	isort sandbox/*
	yapf -ir sandbox/*

format-client:
	mv scripts/client/pyproject.toml scripts/faas/pyproject.toml && yapf -ir scripts/client/* && mv scripts/faas/pyproject.toml scripts/client/pyproject.toml

# mypy --explicit-package-bases sandbox
check:
	pycln --config pyproject.toml --check
	yapf --diff --recursive sandbox/*
	make test
