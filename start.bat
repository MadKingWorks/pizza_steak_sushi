::call "install_dependencies.bat"
set cwd=%~dp0%
echo on
%python_interpreter% setup.py %cwd% 224 10 0.001 40 > start.log
