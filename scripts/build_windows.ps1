<#
.SYNOPSIS
    Builds NukeSim on Windows in one go.

.DESCRIPTION
    1. Finds Visual Studio 2022 and enters its x64 developer environment
       (this puts cl, cmake, ninja and ctest on PATH).
    2. Finds the CUDA Toolkit (CUDA_PATH); falls back to a CPU-only build without it.
    3. Finds or installs vcpkg (C:\vcpkg by default) and sets VCPKG_ROOT.
    4. Configures, builds and runs the unit tests, then prints the GPU info.

    Everything is also written to build\build_windows.log.

.EXAMPLE
    .\build_windows.bat
    .\build_windows.bat -Cpu
    .\build_windows.bat -VcpkgRoot D:\tools\vcpkg
#>
param(
    [switch]$Cpu,
    [switch]$SkipTests,
    [string]$VcpkgRoot = ""
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$userVcpkgRoot = $env:VCPKG_ROOT   # read before the VS shell, which may point it at its own copy

New-Item -ItemType Directory -Force -Path (Join-Path $repo 'build') | Out-Null
$log = Join-Path $repo 'build\build_windows.log'
Set-Content -Path $log -Value "NukeSim build log $(Get-Date -Format s)" -Encoding UTF8

function Step([string]$msg) {
    Write-Host ""
    Write-Host "=== $msg" -ForegroundColor Cyan
    Add-Content -Path $log -Value "`n=== $msg" -Encoding UTF8
}

function Info([string]$msg) {
    Write-Host "    $msg"
    Add-Content -Path $log -Value "    $msg" -Encoding UTF8
}

function Fail([string]$msg) {
    Write-Host ""
    Write-Host "ERROR: $msg" -ForegroundColor Red
    Write-Host "Full log: $log" -ForegroundColor Red
    Add-Content -Path $log -Value "`nERROR: $msg" -Encoding UTF8
    exit 1
}

# Runs a native command; stdout and stderr are shown and logged, a non-zero exit code stops the script.
function Run([string]$exe, [string[]]$arguments) {
    $cmdline = "$exe $($arguments -join ' ')"
    Write-Host "> $cmdline" -ForegroundColor DarkGray
    Add-Content -Path $log -Value "> $cmdline" -Encoding UTF8
    # With 'Stop', Windows PowerShell turns any stderr line of a native tool into a terminating error.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $exe @arguments 2>&1 | ForEach-Object {
        $line = "$_"
        Write-Host $line
        Add-Content -Path $log -Value $line -Encoding UTF8
    }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0) { Fail "command failed with exit code ${code}: $cmdline" }
}

# ---------------------------------------------------------------------------
Step "Visual Studio 2022"
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
    Fail "Visual Studio Installer not found. Install Visual Studio 2022 with the 'Desktop development with C++' workload."
}
$vs = & $vswhere -latest -products * -version '[17.0,18.0)' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) {
    Fail "Visual Studio 2022 with C++ tools not found. In Visual Studio Installer add the 'Desktop development with C++' workload."
}
Info $vs
Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
Set-Location $repo

$cmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmakeCmd) {
    Fail "cmake not found. In Visual Studio Installer enable 'C++ CMake tools for Windows' (or run: winget install Kitware.CMake)."
}
$cmakeVersion = (& cmake --version | Select-Object -First 1)
Info $cmakeVersion
if ($cmakeVersion -match '(\d+)\.(\d+)') {
    if ([int]$Matches[1] -lt 3 -or ([int]$Matches[1] -eq 3 -and [int]$Matches[2] -lt 25)) {
        Fail "CMake 3.25 or newer is required. Update Visual Studio or run: winget install Kitware.CMake"
    }
}
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    Fail "ninja not found. In Visual Studio Installer enable 'C++ CMake tools for Windows'."
}

# ---------------------------------------------------------------------------
Step "CUDA"
$preset = 'windows-ninja'
if ($Cpu) {
    $preset = 'windows-ninja-cpu'
    Info "CPU-only build requested"
} else {
    $nvcc = if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH 'bin\nvcc.exe' } else { '' }
    if (-not $nvcc -or -not (Test-Path $nvcc)) {
        Write-Warning "CUDA Toolkit not found (CUDA_PATH is not set). Building the CPU-only version."
        $preset = 'windows-ninja-cpu'
    } else {
        $env:CUDACXX = $nvcc
        Info ((& $nvcc --version | Select-Object -Last 2) -join ' ')
    }
}

# ---------------------------------------------------------------------------
Step "vcpkg"
$baseline = (Get-Content (Join-Path $repo 'vcpkg.json') -Raw | ConvertFrom-Json).'builtin-baseline'
$root = $null
foreach ($candidate in @($VcpkgRoot, $userVcpkgRoot, 'C:\vcpkg')) {
    if ($candidate -and (Test-Path (Join-Path $candidate 'scripts\buildsystems\vcpkg.cmake'))) {
        $root = $candidate
        break
    }
}
if (-not $root) {
    $root = if ($VcpkgRoot) { $VcpkgRoot } else { 'C:\vcpkg' }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail "git not found; it is needed to install vcpkg." }
    Info "installing vcpkg into $root"
    Run 'git' @('clone', 'https://github.com/microsoft/vcpkg', $root)
}
if (-not (Test-Path (Join-Path $root 'vcpkg.exe'))) {
    Run (Join-Path $root 'bootstrap-vcpkg.bat') @('-disableMetrics')
}
if (Test-Path (Join-Path $root '.git')) {
    # The pinned baseline commit must exist in the local vcpkg clone.
    $ErrorActionPreference = 'Continue'
    & git -C $root cat-file -e $baseline 2>$null
    $haveBaseline = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = 'Stop'
    if (-not $haveBaseline) {
        Info "updating vcpkg (baseline $baseline is missing)"
        Run 'git' @('-C', $root, 'fetch', 'origin')
    }
}
$env:VCPKG_ROOT = $root
[Environment]::SetEnvironmentVariable('VCPKG_ROOT', $root, 'User')
Info "VCPKG_ROOT = $root"

# ---------------------------------------------------------------------------
Step "Configure ($preset)"
Info "The first run builds OpenVDB and its dependencies through vcpkg: 30-60 minutes, once."
Run 'cmake' @('--preset', $preset)

Step "Build"
Run 'cmake' @('--build', '--preset', $preset, '--parallel')

if (-not $SkipTests) {
    Step "Tests"
    Run 'ctest' @('--preset', $preset)
}

Step "Device info"
$exe = Join-Path $repo "build\$preset\nukesim.exe"
Run $exe @('--info')

Write-Host ""
Write-Host "Build OK." -ForegroundColor Green
Write-Host "Next steps (PowerShell, in $repo):"
Write-Host "    .\build\$preset\nukesim.exe scenes\test_fireball.toml"
Write-Host "    .\open_in_blender.bat"
