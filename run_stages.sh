#!/usr/bin/env bash
# Запускає всі етапи послідовно та зберігає stdout у stages_log/.
# stderr (tqdm-прогрес) відкидається, щоб не засмічувати лог.

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

PYTHON=".venv/Scripts/python.exe"
LOG_DIR="stages_log"
mkdir -p "$LOG_DIR"

run_stage() {
    local script="$1"
    local log="$LOG_DIR/$(basename "$script" .py).log"
    local start
    start=$(date '+%Y-%m-%d %H:%M:%S')

    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  Запуск: $script  [$start]"
    echo "════════════════════════════════════════════════════════════════════════════════"

    {
        echo "=== $script | $start ==="
        # 2>/dev/null — відкидаємо stderr (tqdm), у лог іде лише stdout
        "$PYTHON" "$script" 2>/dev/null
        local code=$?
        echo "=== Завершено: $(date '+%Y-%m-%d %H:%M:%S') | exit $code ==="
    } | tee "$log"

    echo "  Лог: $log"
}

run_stage scripts/01_prepare_data.py
run_stage scripts/02_embed.py
run_stage scripts/03_load_to_pinecone.py
run_stage scripts/04_search.py
run_stage scripts/05_chunking.py
run_stage scripts/06_hybrid_search.py

echo ""
echo "Всі етапи завершено. Логи у директорії: $LOG_DIR/"
