@echo off
coverage run --rcfile=.coveragerc -m unittest discover -s test -p "*test.py"
coverage report
coverage html
