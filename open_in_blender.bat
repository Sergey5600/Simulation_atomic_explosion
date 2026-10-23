@echo off
rem Opens a NukeSim output folder in Blender. Options: -Dir out\test_fireball, -Blender path\to\blender.exe, -NoPreviewScene
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\open_in_blender.ps1" %*
