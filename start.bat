call "install_dependencies.bat"
set cwd=%~dp0%
echo on
%python_interpreter% setup.py %cwd% 224 1 0.1 3 > start.log