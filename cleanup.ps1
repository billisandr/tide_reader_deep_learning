# Cleanup script for tide level image processing project
# Removes debug, output, processed, and log directories for clean testing

Write-Host "Starting cleanup of tide level processing directories..." -ForegroundColor Green

$directories = @(
    "data\debug",
    "data\output", 
    "data\processed",
    "logs"
)

foreach ($dir in $directories) {
    if (Test-Path $dir) {
        Write-Host "Clearing contents of directory: $dir" -ForegroundColor Yellow
        Get-ChildItem $dir -Recurse | Remove-Item -Recurse -Force
        Write-Host "Cleared contents of $dir" -ForegroundColor Green
    } else {
        Write-Host "Directory not found: $dir" -ForegroundColor Gray
    }
}

Write-Host "`nCleanup completed! Ready for fresh testing." -ForegroundColor Green
Write-Host "Note: Input and calibration directories were preserved." -ForegroundColor Cyan