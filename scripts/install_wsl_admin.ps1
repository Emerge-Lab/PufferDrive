# Run this file from an elevated Windows PowerShell window.
# It enables WSL2 prerequisites and installs Ubuntu.

$ErrorActionPreference = "Stop"

Write-Host "Enabling Windows Subsystem for Linux..."
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

Write-Host "Enabling Virtual Machine Platform..."
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

Write-Host "Setting WSL 2 as the default version..."
wsl.exe --set-default-version 2

Write-Host "Installing Ubuntu..."
wsl.exe --install -d Ubuntu

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$driveLetter = $repoRoot.Substring(0, 1).ToLowerInvariant()
$wslRepoRoot = "/mnt/$driveLetter/" + $repoRoot.Substring(3).Replace("\", "/")

Write-Host ""
Write-Host "If Windows asks for a restart, reboot, open Ubuntu once to create the Linux user, then run:"
Write-Host "  cd '$wslRepoRoot'"
Write-Host "  bash scripts/wsl_native_3d_setup.sh"
