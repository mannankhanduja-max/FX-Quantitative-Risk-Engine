.PHONY: run run-test run-demo fetch test

# Design-period run (2015-2021). Free to re-run as often as useful.
run:
	python run_fvg_design.py

# Same pipeline on synthetic data, for checking the machinery without
# the real download in place yet.
run-demo:
	python run_fvg_design.py --demo

# Held-out test (2022-now). Refuses until PREREGISTRATION.md is signed
# and committed, and can only run once - see run_fvg_test.py.
run-test:
	python run_fvg_test.py

# Dukascopy XAU/USD 1-minute bid+ask, 2015-now. See FETCH_FVG_DESIGN.md.
fetch:
	python fetch_fvg_raw.py --side both

test:
	pytest tests/ -q
