ROUTES := mujoco311 mujoco4 mjwarp4 newton genesis
MODELS := panda panda_article
RUN = uv run --locked --project envs/$(1) simbench
PORT ?= 8000

.PHONY: sync fetch env-report audit tapes lint test phase1 phase2 phase3 phase4 phase6 \
	plastic mj4new patch terrain tuning versions bench web serve site-data \
	publish-data

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

# Sweep, insertion and deformable ball; flex scenes run only on the MuJoCo routes.
phase6: tapes
	@for r in $(ROUTES); do $(call RUN,$$r) run-plan --route $$r --plan $@ || exit 1; done
	uv run --locked simbench report --plan $@

# MuJoCo 4.0 elasto-plasticity (joint springyield, flex yield) on every route.
plastic:
	@for r in $(ROUTES); do $(call RUN,$$r) run-plan --route $$r --plan $@ || exit 1; done
	uv run --locked simbench report --plan $@

# Speculative contacts (thin card), RK4 orientations (tumbling handle), shell plasticity.
mj4new:
	@for r in $(ROUTES); do $(call RUN,$$r) run-plan --route $$r --plan $@ || exit 1; done
	uv run --locked simbench report --plan $@

# Pinch-and-twist slip torque: point contacts everywhere, hydroelastic patches on Newton.
# G1 walking policy on contact-parameter terrains (needs `make fetch` for the robot).
patch terrain:
	@for r in $(ROUTES); do $(call RUN,$$r) run-plan --route $$r --plan $@ || exit 1; done
	uv run --locked simbench report --plan $@

# 27-candidate push grid per route -> results/summary/tuning/<route>.json
tuning: tapes
	@for r in $(ROUTES); do $(call RUN,$$r) tune --route $$r || exit 1; done

# Takes about 15 minutes; run with the GPU otherwise idle.
bench:
	@for r in $(ROUTES); do $(call RUN,$$r) bench --route $$r || exit 1; done

web:
	$(call RUN,mujoco4) export-web

# GitHub Pages: web/ is deployed by .github/workflows/pages.yml with the data from the
# `site-data` release. The public export leaves out the article's own traces.
site-data:
	$(call RUN,mujoco4) export-web --public
	mkdir -p dist
	tar -czf dist/site-data.tar.gz -C web data
	@du -h dist/site-data.tar.gz

publish-data: site-data
	gh release view site-data >/dev/null 2>&1 || gh release create site-data --latest=false \
		--title "Site data" --notes "Generated web/data for GitHub Pages (make publish-data)."
	gh release upload site-data dist/site-data.tar.gz --clobber
	gh workflow run pages.yml
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
