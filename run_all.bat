@echo off
rem Runs the three cinematic simulations one after another (fireball, mushroom, sand).
rem Options: -Out G:\nukesim_out, -Only mushroom,sand, -Force (run even if the disk looks too small)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_all.ps1" %*
