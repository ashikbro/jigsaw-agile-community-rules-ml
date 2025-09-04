Write-Host "Starting API server..."
$ErrorActionPreference = "Stop"

try {
    # Check if Python is available
    $pythonVersion = python --version
    Write-Host "Using Python version: $pythonVersion"
    
    # Check if required packages are installed
    Write-Host "Checking required packages..."
    pip install -r requirements_simple.txt
    
    # Start the API server
    Write-Host "Starting the API server..."
    Write-Host "API will be available at: http://localhost:8000"
    Write-Host "Press Ctrl+C to stop the server"
    
    # Run the API
    python -m uvicorn simple_api:app --host 127.0.0.1 --port 8000 --reload
} catch {
    Write-Host "An error occurred: $_" -ForegroundColor Red
    Write-Host "Make sure Python is installed and in your PATH"
    exit 1
}
