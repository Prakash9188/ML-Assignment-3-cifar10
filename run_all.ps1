# Reproduce everything: python run_all steps in order (each script is resumable via cached runs).
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
foreach ($q in "q1", "q2", "q3", "q4") {
    python -u "$q.py" *> "logs\$q.log"
}
