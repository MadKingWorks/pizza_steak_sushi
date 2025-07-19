::call "install_dependencies.bat"
set cwd=%~dp0%
echo on
%python_interpreter% setup.py %cwd% 224 10 0.1 4 > start.log
