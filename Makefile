VENV = env
PYTHON = $(VENV)/Scripts/python
PIP = $(VENV)/Scripts/pip

setup: requirements.txt
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt

# activate environment: https://docs.python.org/3/tutorial/venv.html

clean:
	rm -rf __pycache__
	rm -rf $(VENV)