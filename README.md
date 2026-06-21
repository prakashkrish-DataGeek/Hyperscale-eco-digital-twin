# The Hyperscale Bio-Digital Twin

**Geospatial Climate Risk & Thermal Integrity Engine for AI Data Centers**

A production-style Python prototype that treats a global AI data centre fleet as a living system: it fuses macro climate stress signals with asset telemetry, predicts next-hour thermal efficiency degradation with machine learning, and prescribes sustainability-aware workload redistribution — all inside a dark, mission-control Streamlit experience.

> *AI data centre growth is increasingly constrained not only by compute and power, but by thermal physics, water availability, carbon intensity, and regional climate volatility.*

---

## Why this exists

The next constraint on AI infrastructure is ecological, and it is already measurable. Wet-bulb temperature caps how much free and evaporative cooling a campus can extract from the atmosphere. Watershed stress limits evaporative headroom and licence-to-operate. Grid carbon intensity prices every marginal megawatt in emissions. This prototype demonstrates how those signals can be unified into a single operational decision: **where should the next megawatt of AI training run, right now?**

## What it does

The twin monitors four strategic AI regions — Northern Virginia, West Texas, Dublin, and Frankfurt — each with a distinct climate and infrastructure personality, and answers five executive questions:

1. Which region is under the greatest environmental and operational stress?
2. Which facility is most thermally efficient?
3. Where is next-hour PUE degradation most likely?
4. How should AI workloads be redistributed to cut risk, emissions, and water draw?
5. What strategic signal should an executive act on immediately?

## Quick start

```bash
git clone <your-repo-url>
cd hyperscale-bio-digital-twin
pip install -r requirements.txt
streamlit run app.py
```

No external APIs, credentials, or internet access required. All data is synthetic, deterministic (seeded), and generated on first launch in a few seconds.

## Architecture

```
┌────────────────────┐     ┌────────────────────────┐     ┌──────────────────────┐
│  data_generator.py │ ──> │   analytics_engine.py  │ ──> │        app.py        │
│  DataCenterSimulator     │   BioDigitalTwinEngine │     │  Streamlit control   │
│                    │     │                        │     │  room (cached)       │
│  30d x 4 regions   │     │  features -> GBM model │     │  KPIs / pydeck map / │
│  1h coupled climate│     │  -> risk -> optimiser  │     │  drill-down / ML /   │
│  + plant telemetry │     │                        │     │  prescriptive view   │
└────────────────────┘     └────────────────────────┘     └──────────────────────┘
```

### `data_generator.py` — the simulated world

`DataCenterSimulator` produces a 30-day, 1-hour resolution telemetry stream (~2,880 region-hours) built from three coupled subsystems:

- **Climate** — diurnal and synoptic cycles, region-specific baselines and volatility, scripted heat-wave episodes for West Texas and Frankfurt. Wet-bulb temperature is computed explicitly from temperature and humidity using the Stull (2011) approximation.
- **Compute** — weekday/weekend rhythm plus scheduled AI training surge windows that mimic real LLM campaign behaviour.
- **Thermal plant** — chiller power derived from IT heat rejection multiplied by a non-linear wet-bulb difficulty curve, passed through a first-order lag to model the thermal inertia of chilled-water loops. PUE and WUE are *derived* quantities, never sampled independently, so the physics stays internally consistent.

Defensive assertions guarantee PUE stays in a plausible band, wet-bulb never exceeds dry-bulb, and no NaNs escape the generator.

### `analytics_engine.py` — prediction and prescription

`BioDigitalTwinEngine` provides four layers:

- **Feature engineering** — an Environmental Stress Index (wet-bulb 50%, watershed stress 30%, grid carbon 20%, all min-max normalised over the fleet), 1-hour and 3-hour lags, 6-hour rolling means, calendar features, and the `next_hour_pue` target with its `pue_degradation` form. All transforms are grouped per region so nothing leaks across regional boundaries.
- **Predictive model** — a `GradientBoostingRegressor` trained on a strictly chronological split (the model is always evaluated on the future, never on shuffled leakage), reporting RMSE, MAE, R², and feature importances.
- **Risk classification** — predicted PUE degradation mapped to Low / Moderate / High bands. A +0.01 PUE delta on a 100 MW campus is roughly 1 MW of pure overhead, so even small deltas are material.
- **Prescriptive optimiser** — `optimize_workload_distribution()` ranks regions by a transparent placement cost (stress + predicted degradation + carbon), shifts up to 20% of the worst region's load to the two best, and quantifies avoided CO₂ (carbon-intensity delta including PUE overhead) and avoided water (donor evaporative intensity weighted by basin stress). Deliberately greedy and explainable rather than a black-box solver — every recommendation survives a one-sentence explanation to a CTO.

### `app.py` — the control room

A dark, glass-panel Streamlit interface with custom CSS: an executive KPI row, a pydeck global map with stress-coloured glowing markers, a regional drill-down with a dual-axis wet-bulb vs chiller chart and high-risk shading, an ML intelligence view, and a prescriptive orchestration directive that reads like an operations command. Heavy work (simulation, training, scoring) is cached with `st.cache_data` / `st.cache_resource`, so the UI stays instant after first load.

## The four regions

| Region | Personality | Key constraint |
|---|---|---|
| Northern Virginia | High humidity, congested PJM grid, heavy AI utilisation | Humid wet-bulb stress |
| West Texas | Extreme dry heat, deep solar carbon dips, severe water scarcity | Water + thermal volatility |
| Dublin | Cool marine climate, strong sustainability pressure | Planning / licence-to-operate |
| Frankfurt | Dense industrial grid, green-power limits, episodic heat waves | Grid + heat-wave sensitivity |

## Design decisions worth noting

- **Determinism everywhere.** A single seeded NumPy generator drives the world; the model and optimiser are seeded too. Every run reproduces the same state — essential for a portfolio demo and good practice for any simulation.
- **Physics before pixels.** PUE responds to wet-bulb through a non-linear difficulty curve with lag, so the ML model has genuine structure to learn rather than noise dressed up as signal.
- **Time-honest evaluation.** Chronological train/test split only. Shuffled splits flatter time-series models and would be professional malpractice here.
- **Transparency over cleverness.** The optimiser is a greedy marginal-cost re-allocation by design. In an executive setting, an explainable 90% answer beats an opaque 99% one.

## Limitations and honest caveats

This is a prototype built on synthetic data. The climate profiles are stylised (anchored in public domain knowledge, not measured feeds), the water and carbon savings are directional estimates, and the optimiser ignores network egress, data gravity, and contractual capacity constraints that a production system would have to respect. The architecture, however, is the point: swap the simulator for live telemetry feeds and the rest of the pipeline stands.

## Roadmap

- Live data adapters (NOAA/ERA5 weather, WattTime/Electricity Maps carbon, facility BMS telemetry)
- Probabilistic forecasts (quantile regression) instead of point estimates
- Multi-hour optimisation horizon with workload migration costs
- Scenario stress-testing: 2030 climate projections per region

## Licence

MIT — use it, fork it, build on it.

---

*Built as a flagship portfolio piece spanning geospatial digital twins, climate-aware operations, MLOps prototyping, and industrial AI storytelling.*
