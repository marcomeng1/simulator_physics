ROUTES := mujoco311 mujoco4 mjwarp4 newton genesis
MODELS := panda panda_article
RUN = uv run --locked --project envs/$(1) simbench
PORT ?= 8000

.PHONY: sync fetch env-report audit tapes lint test phase1 phase2 phase3 phase4 versions \
	bench web serve

phase1 phase2:
	@for r in $(ROUTES); do $(call RUN,$$r) run-plan --route $$r --plan $@ || exit 1; done
	uv run --locked simbench report --plan $@

# The Panda tapes are generated once on MuJoCo 3.11, as in the article.
tapes:
	$(call RUN,mujoco311) make-tapes

phase3: tapes
	@for r in $(ROUTES); do $(call RUN,$$r) run-plan --route $$r --plan $@ || exit 1; done
	uv run --locked simbench report --plan $@

# MuJoCo 3.11 vs 4.0 on every phase 1-3 variant (needs phase1-3 traces).
versions:
	uv run --locked simbench report --plan versions

phase4: tapes
	@for r in $(ROUTES); do $(call RUN,$$r) run-plan --route $$r --plan $@ || exit 1; done
	uv run --locked simbench report --plan $@

# Takes about 15 minutes; run with the GPU otherwise idle.
bench:
	@for r in $(ROUTES); do $(call RUN,$$r) bench --route $$r || exit 1; done

web:
	$(call RUN,mujoco4) export-web

serve:
	python3 -m http.server $(PORT) -d web

sync:
	uv sync --locked
	@for r in $(ROUTES); do uv sync --locked --project envs/$$r || exit 1; done

fetch:
	uv run --locked simbench fetch

env-report:
	@for r in $(ROUTES); do $(call RUN,$$r) env-report --route $$r || exit 1; done

audit:
	@for m in $(MODELS); do \
	  for r in $(ROUTES); do $(call RUN,$$r) audit --route $$r --model $$m || exit 1; done; \
	  uv run --locked simbench audit-compare --model $$m; \
	done

lint:
	uv run --locked ruff check .

test:
	uv run --locked pytest
