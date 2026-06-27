#!/bin/bash

set -e

REPORT_DIR="reports"

mkdir -p "$REPORT_DIR"

echo "=============================================="
echo " Customer Churn - Quality Pipeline"
echo "=============================================="

echo ""
echo "[1/5] Black - Code formatting"

black . \
    --exclude='/(venv|__pycache__|reports)/' \
    > "$REPORT_DIR/black.txt" 2>&1 || true


echo ""
echo "[2/5] Flake8 - Code style"

flake8 . \
    --exclude=venv,__pycache__,reports \
    > "$REPORT_DIR/flake8.txt" 2>&1 || true


echo ""
echo "[3/5] Pylint - Code quality"

pylint \
    main.py \
    data_loader.py \
    trainer.py \
    evaluator.py \
    model_io.py \
    > "$REPORT_DIR/pylint.txt" 2>&1 || true


echo ""
echo "[4/5] Bandit - Security"

bandit \
    -r . \
    -x ./venv,./__pycache__,./reports \
    > "$REPORT_DIR/bandit.txt" 2>&1 || true


echo ""
echo "[5/5] Pytest - Tests"

pytest \
    --junitxml="$REPORT_DIR/pytest.xml" \
    > "$REPORT_DIR/pytest.txt" 2>&1 || true


echo ""
echo "=============================================="
echo " Quality check finished"
echo " Reports available in $REPORT_DIR"
echo "=============================================="

ls -lh "$REPORT_DIR"
