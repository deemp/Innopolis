VENV = env
PYTHON = $(VENV)/Scripts/python3
PIP = $(VENV)/Scripts/pip3

setup: $(VENV)/bin/activate
	$(PYTHON) main.py

$(VENV)/bin/activate: requirements.txt
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt

clean:
	rm -rf __pycache__
	rm -rf $(VENV)