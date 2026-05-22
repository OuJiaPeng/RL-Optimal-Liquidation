# RL Optimal Liquidation — convenience targets.
# Each train target runs a single seed; for multi-seed sweeps see scripts/.

.PHONY: install test phase1 phase2-vol phase2-spread diagnose oracle reevaluate clean help

SEED ?= 0

help:
	@echo "Targets:"
	@echo "  install         pip install -e .[dev]"
	@echo "  test            run the test suite (17 tests, ~0.2s)"
	@echo "  phase1          train Phase 1 (linear-impact AC validation)        ~5 min"
	@echo "  phase2-vol      train Phase 2 vol-conditioning (positive result)   ~5 min"
	@echo "  phase2-spread   train Phase 2 spread-conditioning (mixed result)   ~5 min"
	@echo "  diagnose        run diagnose.py on the most-recent phase1 model"
	@echo "  oracle          compute matched-pair oracle baselines"
	@echo "  reevaluate      re-score existing best_model checkpoints under textbook-AC env"
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

phase2-spread:
	python scripts/train_ppo.py --config configs/phase2_spread.yaml --seed $(SEED) --output runs/phase2_spread_s$(SEED)

diagnose:
	python scripts/diagnose.py --config configs/phase1.yaml --model runs/phase1_s$(SEED)/best_model.zip --output runs/phase1_s$(SEED)/diagnostics

oracle:
	@echo "--- Vol oracle (noise=0.3) ---"
	python scripts/probe_vol_oracle.py --noise-std 0.3
	@echo ""
	@echo "--- Spread oracle (noise=2.0) ---"
	python scripts/probe_eta_oracle.py --eta-noise-std 2.0

reevaluate:
	python scripts/reevaluate_after_textbook_ac.py

clean:
	rm -rf runs/
