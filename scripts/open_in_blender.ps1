<#
.SYNOPSIS
    Opens a NukeSim output folder in Blender via tools\blender\import_explosion.py.

.EXAMPLE
    .\open_in_blender.bat
    .\open_in_blender.bat -Shots                      (fireball + mushroom + sand from out\)
    .\open_in_blender.bat -Dir out\mushroom_20kt -Extra "--glow 2 --detail 0.3"
    .\open_in_blender.bat -Dir out\test_fireball -NoPreviewScene
    .\open_in_blender.bat -Blender "D:\Blender 5.2\blender.exe"
#>
param(
    [string]$Dir = "out\test_fireball",
    [string]$Blender = "",
    [switch]$NoPreviewScene,
    [switch]$Shots,           # import out\fireball_closeup, out\mushroom_20kt and out\sand_20kt together
    [string[]]$Extra = @()    # more import_explosion.py options, e.g. -Extra "--glow 2 --detail 0.3"
)

$ErrorActionPreference = 'Stop'
# Through the .bat the options arrive as one string: split it into separate arguments.
$Extra = @($Extra | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Find-Blender {
    $cmd = Get-Command blender -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $roots = @(
        (Join-Path $env:ProgramFiles 'Blender Foundation'),
        (Join-Path ${env:ProgramFiles(x86)} 'Steam\steamapps\common')
    )
    $found = foreach ($r in $roots) {
        if (Test-Path $r) { Get-ChildItem -Path $r -Filter 'blender.exe' -Recurse -Depth 2 -ErrorAction SilentlyContinue }
    }
    # Prefer the newest version folder, e.g. "Blender 5.2".
    $best = $found | Sort-Object { if ($_.DirectoryName -match '(\d+)\.(\d+)') { [version]"$($Matches[1]).$($Matches[2])" } else { [version]'0.0' } } -Descending |
        Select-Object -First 1
    if ($best) { return $best.FullName }
    return $null
}

if (-not $Blender) { $Blender = Find-Blender }
if (-not $Blender -or -not (Test-Path $Blender)) {
    Write-Host "Blender not found. Pass its path: .\open_in_blender.bat -Blender `"C:\path\to\blender.exe`"" -ForegroundColor Red
    exit 1
}
if ($Shots) {
    $script = Join-Path $repo 'tools\blender\import_shots.py'
    $blenderArgs = @('--python', $script, '--', '--root', (Join-Path $repo 'out'))
    if (-not $NoPreviewScene) { $blenderArgs += '--preview-scene' }
    $blenderArgs += $Extra
    Write-Host "Starting $Blender"
    & $Blender @blenderArgs
    exit 0
}
$outDir = Resolve-Path $Dir -ErrorAction SilentlyContinue
if (-not $outDir -or -not (Test-Path (Join-Path $outDir 'manifest.json'))) {
    Write-Host "No manifest.json in '$Dir'. Run the simulation first, e.g.:" -ForegroundColor Red
    Write-Host "    .\build\windows-ninja\nukesim.exe scenes\test_fireball.toml"
    exit 1
}

$script = Join-Path $repo 'tools\blender\import_explosion.py'
$blenderArgs = @('--python', $script, '--', '--dir', $outDir.Path)
if (-not $NoPreviewScene) { $blenderArgs += '--preview-scene' }
$blenderArgs += $Extra
Write-Host "Starting $Blender"
& $Blender @blenderArgs
