@echo off
setlocal

if "%SPHINXBUILD%" == "" set SPHINXBUILD=sphinx-build
if "%SPHINXOPTS%" == "" set SPHINXOPTS=
if "%SOURCEDIR%" == "" set SOURCEDIR=.
if "%BUILDDIR%" == "" set BUILDDIR=_build

%SPHINXBUILD% -M %1 "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%

:end
