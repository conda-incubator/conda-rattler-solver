set CONDA_SOLVER=rattler
conda create --dry-run scipy
if errorlevel 1 exit /b 1
