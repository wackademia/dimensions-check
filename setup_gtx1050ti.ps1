param(
    [ValidateSet('3.11', '3.12')][string]$PythonVersion = '3.11'
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
            throw 'Install 64-bit Python 3.11 with the Python launcher, then run this script again.'
        }
        & py "-$PythonVersion" -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv. Install Python 3.11 or pass -PythonVersion 3.12.' }
    }
    $ProjectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    & $ProjectPython -c "import struct,sys; assert sys.version_info[:2] in ((3,11),(3,12)) and struct.calcsize('P')==8, 'Use 64-bit Python 3.11 or 3.12'"
    if ($LASTEXITCODE -ne 0) { throw 'The existing .venv is incompatible. See docs/GTX1050TI_SETUP.md.' }
    & $ProjectPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
    & $ProjectPython -m pip install 'torch==2.7.1+cu118' 'torchvision==0.22.1+cu118' --index-url https://download.pytorch.org/whl/cu118
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed. Check your internet connection and Python version.' }
    & $ProjectPython -m pip install -r requirements-gtx1050ti.txt -c requirements-gtx1050ti.txt -e .
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $ProjectPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
    & $ProjectPython scripts/download_models.py
    if ($LASTEXITCODE -ne 0) { throw 'Model download or checksum verification failed. Re-run setup to retry.' }
    & $ProjectPython scripts/check_runtime.py --require-cuda
    if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed. See the setup guide for driver/wheel troubleshooting.' }
    Write-Host 'Setup complete. Run .\run_static_demo.bat and open http://127.0.0.1:8080'
} finally {
    Pop-Location
}
