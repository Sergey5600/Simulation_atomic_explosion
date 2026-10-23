@echo off
rem Builds NukeSim: finds Visual Studio, installs vcpkg if needed, builds and runs the tests.
rem Options: -Cpu (no CUDA), -SkipTests, -VcpkgRoot D:\path\to\vcpkg
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build_windows.ps1" %*
