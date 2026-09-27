# Convenience shortcuts. `make all` runs the whole pipeline.
PY = python

.PHONY: all baseline synth train maven apply validate figures test clean

all:
	$(PY) scripts/run_all.py

baseline:
	$(PY) scripts/01_build_classical_baseline.py

synth:
	$(PY) scripts/02_generate_synthetic.py

train:
	$(PY) scripts/03_train_pinn.py

maven:
	$(PY) scripts/04_maven_pretrain.py

apply:
	$(PY) scripts/05_apply_and_recover.py

validate:
	$(PY) scripts/06_validate_hop.py

figures:
	$(PY) scripts/07_make_figures.py

test:
	pytest -q tests/

clean:
	rm -rf outputs
