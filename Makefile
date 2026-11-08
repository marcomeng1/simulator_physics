ROUTES := mujoco311 mujoco4 mjwarp4 newton genesis
MODELS := panda panda_article
RUN = uv run --locked --project envs/$(1) simbench

.PHONY: sync fetch env-report audit lint test

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
