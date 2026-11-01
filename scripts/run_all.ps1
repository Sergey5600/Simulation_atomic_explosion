<#
.SYNOPSIS
    Runs the three cinematic simulations one after another: fireball close-up, mushroom, sand.

.DESCRIPTION
    Before each simulation it checks the free space on the output disk against what that
    simulation writes and skips it (with a red message) when it would not fit, so a night is
    not lost on a run that stops half way. Each run is logged to <out>\<scene>.log and a
    summary is printed at the end. nukesim itself also stops cleanly before the disk is full.

.EXAMPLE
    .\run_all.bat
    .\run_all.bat -Out G:\nukesim_out              (all three into G:\nukesim_out\<scene>)
    .\run_all.bat -Only mushroom,sand
#>
param(
    [string]$Out = "",                 # parent folder of the outputs (default: out\ in the project)
    [string[]]$Only = @(),             # subset: fireball, mushroom, sand
    [switch]$Force,                    # run even when the free space looks too small
    [string]$Exe = "build\windows-ninja\nukesim.exe"
)

$ErrorActionPreference = 'Stop'
# Through the .bat the list arrives as one string: split it.
$Only = @($Only | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# Order matters: the close-up first (it is the one that has to be redone), the sand last.
$shots = @(
    [pscustomobject]@{ Key = 'fireball'; Scene = 'scenes\fireball_closeup.toml'; Folder = 'fireball_closeup'; Gb = 175 },
    [pscustomobject]@{ Key = 'mushroom'; Scene = 'scenes\mushroom_20kt.toml';   Folder = 'mushroom_20kt';    Gb = 110 },
    [pscustomobject]@{ Key = 'sand';     Scene = 'scenes\sand_20kt.toml';       Folder = 'sand_20kt';        Gb = 35 }
)
if ($Only.Count -gt 0) {
    foreach ($o in $Only) {
        if (-not ($shots | Where-Object { $_.Key -eq $o })) { throw "unknown -Only entry '$o' (use fireball, mushroom, sand)" }
    }
    $shots = @($shots | Where-Object { $Only -contains $_.Key })
}

if (-not (Test-Path $Exe)) {
    Write-Host "$Exe not found: build first with .\build_windows.bat" -ForegroundColor Red
    exit 1
}
$Exe = (Resolve-Path $Exe).Path
$parent = if ($Out) { $Out } else { Join-Path $repo 'out' }
New-Item -ItemType Directory -Force -Path $parent | Out-Null
$parent = (Resolve-Path $parent).Path

function Get-FreeGb([string]$path) {
    $root = [System.IO.Path]::GetPathRoot($path)
    return [math]::Floor((New-Object System.IO.DriveInfo($root)).AvailableFreeSpace / 1GB)
}

$total = [math]::Round(($shots | Measure-Object -Property Gb -Sum).Sum)
Write-Host ("Running {0} simulation(s) into {1}: about {2} GB in all, {3} GB free now" -f $shots.Count, $parent, $total, (Get-FreeGb $parent))
Write-Host "Close Blender, games and the browser while it runs: they take GPU memory."

$results = @()
foreach ($s in $shots) {
    $dir = Join-Path $parent $s.Folder
    # Frames of an earlier run in the same folder are overwritten, so their space counts as free.
    $old = 0
    if (Test-Path $dir) {
        $old = [math]::Floor(((Get-ChildItem $dir -Filter *.vdb -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum) / 1GB)
    }
    $free = (Get-FreeGb $parent) + $old
    if ($free -lt $s.Gb -and -not $Force) {
        Write-Host ("SKIPPED {0}: needs about {1} GB, only {2} GB available (free some space or use -Out on another disk)" -f $s.Key, $s.Gb, $free) -ForegroundColor Red
        $results += [pscustomobject]@{ Simulation = $s.Key; Result = 'skipped (disk)'; Time = '-'; Folder = $dir }
        continue
    }
    $log = Join-Path $parent ($s.Folder + '.log')
    Write-Host ""
    Write-Host ("=== {0}: {1} -> {2} (log: {3})" -f $s.Key, $s.Scene, $dir, $log) -ForegroundColor Cyan
    $start = Get-Date
    # Error lines arrive on stderr: keep them as plain text in the log instead of stopping the script.
    $ErrorActionPreference = 'Continue'
    & $Exe $s.Scene --out $dir 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    $span = (Get-Date) - $start
    $time = '{0}h {1:00}m' -f [math]::Floor($span.TotalHours), $span.Minutes
    $result = if ($code -eq 0) { 'done' } else { "FAILED (exit $code, see the log)" }
    $results += [pscustomobject]@{ Simulation = $s.Key; Result = $result; Time = $time; Folder = $dir }
}

Write-Host ""
$results | Format-Table -AutoSize
if ($results | Where-Object { $_.Result -ne 'done' }) { exit 1 }
