# https://earthly.dev/blog/python-makefile/

VENV = env
PYTHON = $(VENV)/Scripts/python
PIP = $(VENV)/Scripts/pip3
BIG_1 = Big-HW-1/Task/main.py
HW_2_1 = HW2/Task-1/main.py
TARGET = HW_2_1

# run Task

run: $(VENV)/Scripts/activate
	$(PYTHON) $($(TARGET))

$(VENV)/Scripts/activate: requirements.txt
# create environment
	python -m venv $(VENV)
# install requirements
# $(PIP) install -r requirements.txt

clean:
	python -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	python -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"