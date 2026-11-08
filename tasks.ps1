param (
    [Parameter(Position=0)]
    [string]$Task = "help"
)

switch ($Task.ToLower()) {
    "run" {
        Write-Host "Starting FusionRAG server on http://localhost:8000..." -ForegroundColor Cyan
        python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    }
    "test" {
        Write-Host "Running pytest test suite..." -ForegroundColor Cyan
        python -m pytest tests/ -v
    }
    "seed" {
        Write-Host "Seeding technical knowledge base..." -ForegroundColor Cyan
        python scripts/seed_data.py
    }
    "benchmark" {
        Write-Host "Running latency & accuracy benchmark..." -ForegroundColor Cyan
        python scripts/benchmark.py
    }
    "evaluate" {
        Write-Host "Running RAG Triad automated evaluation..." -ForegroundColor Cyan
        python scripts/evaluate_rag.py
    }
    "cli" {
        Write-Host "Launching interactive CLI terminal..." -ForegroundColor Cyan
        python scripts/cli_chat.py
    }
    default {
        Write-Host "FusionRAG Task Runner Commands:" -ForegroundColor Yellow
        Write-Host "  .\tasks.ps1 run        - Start FastAPI server"
        Write-Host "  .\tasks.ps1 test       - Run 22 test cases"
        Write-Host "  .\tasks.ps1 seed       - Seed sample technical corpus"
        Write-Host "  .\tasks.ps1 benchmark  - Run latency/accuracy benchmarks"
        Write-Host "  .\tasks.ps1 evaluate   - Run RAG Triad quality evaluation"
        Write-Host "  .\tasks.ps1 cli        - Open rich interactive terminal chat"
    }
}
