$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
python -B -m unittest discover -v
python -B -m tests.retrieval_benchmark
