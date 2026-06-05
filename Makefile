# RL Optimal Liquidation — convenience targets.
# Each train target runs a single seed; for multi-seed sweeps see scripts/.

.PHONY: install test phase1 phase2-vol diagnose oracle clean help

SEED ?= 0

help:
	@echo "Targets:"
	@echo "  install         pip install -e .[dev]"
	@echo "  test            run the test suite"
	@echo "  phase1          train Phase 1 (linear-impact AC validation)        ~5 min"
	@echo "  phase2-vol      train Phase 2 vol-conditioning (positive result)   ~5 min"
	@echo "  diagnose        run diagnose.py on the most-recent phase1 model"
	@echo "  oracle          compute matched-pair oracle baselines"
	@echo "  clean           wipe runs/ (DESTRUCTIVE)"
	@echo ""
	@echo "Override seed:  make phase1 SEED=3"

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

phase1:
	python scripts/train_ppo.py --config configs/phase1.yaml --seed $(SEED) --output runs/phase1_s$(SEED)

phase2-vol:
	python scripts/train_ppo.py --config configs/phase2_vol.yaml --seed $(SEED) --output runs/phase2_vol_s$(SEED)

diagnose:
	python scripts/diagnose.py --config configs/phase1.yaml --model runs/phase1_s$(SEED)/best_model.zip --output runs/phase1_s$(SEED)/diagnostics

oracle:
	@echo "--- Vol oracle (noise=0.3) ---"
	python scripts/probe_vol_oracle.py --noise-std 0.3

clean:
	rm -rf runs/
