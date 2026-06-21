"""
analytics_engine.py
===================

Predictive and prescriptive intelligence layer of the Hyperscale
Bio-Digital Twin.

Pipeline
--------
raw telemetry
    -> engineered features (stress index, lags, rolling means)
    -> time-ordered ML regression (next-hour PUE)
    -> interpretable risk classification (Low / Moderate / High)
    -> deterministic workload redistribution optimiser
       with CO2 and water-savings estimates.

Design choices
--------------
* GradientBoostingRegressor: strong tabular performance, deterministic
  with a fixed seed, and yields clean feature importances for the
  executive 'what drives risk' narrative.
* The train/test split is strictly chronological - the model is always
  evaluated on the future, never on shuffled leakage.
* The optimiser is intentionally transparent: a greedy marginal-cost
  re-allocation rather than a black-box solver, so every recommendation
  can be explained in one sentence to a CTO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_SEED: int = 42

# ----------------------------------------------------------------------
# Environmental Stress Index weights.
#
# Rationale: wet-bulb temperature is the dominant physical constraint on
# cooling (50%), watershed stress constrains evaporative headroom and
# licence-to-operate (30%), and grid carbon intensity captures the
# sustainability cost of every additional megawatt (20%). Each component
# is min-max normalised to [0, 1] over the observed fleet so the index
# is comparable across regions.
# ----------------------------------------------------------------------
ESI_WEIGHTS: Dict[str, float] = {
    "wet_bulb_temp_c": 0.50,
    "watershed_stress_index": 0.30,
    "grid_carbon_intensity_gco2_kwh": 0.20,
}

# Risk thresholds on predicted next-hour PUE degradation (delta PUE).
# +0.01 PUE on a 100 MW campus is ~1 MW of pure overhead, so even small
# deltas are operationally material.
RISK_THRESHOLDS: Tuple[float, float] = (0.005, 0.02)  # low/moderate, moderate/high

# Emission and water factors used by the optimiser.
HOURS_PER_SHIFT_WINDOW: float = 4.0   # recommendation covers a 4-hour window
GPM_TO_LITRES_PER_HOUR: float = 3.785 * 60.0


@dataclass
class ModelReport:
    """Container for model evaluation artefacts."""

    rmse: float
    mae: float
    r2: float
    n_train: int
    n_test: int
    feature_importances: pd.Series


@dataclass
class WorkloadShift:
    """One recommended movement of AI load between regions."""

    source: str
    destination: str
    shift_mw: float
    shift_pct_of_source: float
    co2_saved_kg: float
    water_saved_litres: float
    rationale: str


class BioDigitalTwinEngine:
    """Transforms telemetry into prediction, risk, and prescription."""

    LAG_FEATURES = ["wet_bulb_temp_c", "it_compute_load_mw",
                    "chiller_power_mw", "pue"]
    ROLLING_FEATURES = ["wet_bulb_temp_c", "it_compute_load_mw"]

    def __init__(self, telemetry: pd.DataFrame,
                 seed: int = RANDOM_SEED) -> None:
        if telemetry is None or telemetry.empty:
            raise ValueError("telemetry DataFrame must be non-empty")
        required = {"timestamp", "region", "pue", "wet_bulb_temp_c",
                    "watershed_stress_index",
                    "grid_carbon_intensity_gco2_kwh"}
        missing = required - set(telemetry.columns)
        if missing:
            raise ValueError(f"telemetry missing columns: {sorted(missing)}")

        self.seed = seed
        self.raw = telemetry.copy()
        self.features: Optional[pd.DataFrame] = None
        self.model: Optional[GradientBoostingRegressor] = None
        self.report: Optional[ModelReport] = None
        self.feature_columns: List[str] = []

    # ------------------------------------------------------------------
    # A. Feature engineering
    # ------------------------------------------------------------------
    def engineer_features(self) -> pd.DataFrame:
        """Build the modelling table: stress index, lags, rolling means,
        and the next-hour PUE target. All groupbys are per-region so no
        feature ever leaks across regional boundaries."""
        df = self.raw.sort_values(["region", "timestamp"]).copy()

        # Environmental Stress Index - normalise each component over the
        # whole fleet, then apply the explicit weights documented above.
        for col in ESI_WEIGHTS:
            lo, hi = df[col].min(), df[col].max()
            span = (hi - lo) if hi > lo else 1.0
            df[f"_norm_{col}"] = (df[col] - lo) / span
        df["environmental_stress_index"] = sum(
            w * df[f"_norm_{c}"] for c, w in ESI_WEIGHTS.items()
        )
        df = df.drop(columns=[f"_norm_{c}" for c in ESI_WEIGHTS])

        grouped = df.groupby("region", sort=False)

        # Lag features: the plant's thermal inertia means the recent past
        # is highly predictive of the next hour.
        for col in self.LAG_FEATURES:
            df[f"{col}_lag1"] = grouped[col].shift(1)
            df[f"{col}_lag3"] = grouped[col].shift(3)

        # Rolling means smooth weather noise into trend signals.
        for col in self.ROLLING_FEATURES:
            df[f"{col}_roll6"] = (
                grouped[col].transform(
                    lambda s: s.rolling(6, min_periods=3).mean())
            )

        # Calendar features capture diurnal/weekly load structure.
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek

        # Prediction target and its degradation form.
        df["next_hour_pue"] = grouped["pue"].shift(-1)
        df["pue_degradation"] = df["next_hour_pue"] - df["pue"]

        self.features = df
        return df

    # ------------------------------------------------------------------
    # B. Predictive ML pipeline
    # ------------------------------------------------------------------
    def _feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Explicit, ordered feature list used for both fit and inference."""
        self.feature_columns = [
            "environmental_stress_index",
            "wet_bulb_temp_c", "relative_humidity_pct", "ambient_temp_c",
            "watershed_stress_index", "grid_carbon_intensity_gco2_kwh",
            "it_compute_load_mw", "chiller_power_mw", "pue",
            "wet_bulb_temp_c_lag1", "wet_bulb_temp_c_lag3",
            "it_compute_load_mw_lag1", "it_compute_load_mw_lag3",
            "chiller_power_mw_lag1", "chiller_power_mw_lag3",
            "pue_lag1", "pue_lag3",
            "wet_bulb_temp_c_roll6", "it_compute_load_mw_roll6",
            "hour_of_day", "day_of_week",
        ]
        return df[self.feature_columns]

    def train_model(self, test_fraction: float = 0.2) -> ModelReport:
        """Chronological train/test split + gradient boosting fit.

        The split point is a timestamp quantile: everything before it
        trains the model, everything after evaluates it. No shuffling.
        """
        if self.features is None:
            self.engineer_features()
        df = self.features.dropna(
            subset=["next_hour_pue"] + [
                c for c in self.features.columns
                if c.endswith(("lag1", "lag3", "roll6"))
            ]
        ).copy()

        cutoff = df["timestamp"].quantile(1.0 - test_fraction)
        train, test = df[df["timestamp"] <= cutoff], df[df["timestamp"] > cutoff]
        if train.empty or test.empty:
            raise RuntimeError("Chronological split produced an empty set")

        x_train = self._feature_matrix(train)
        x_test = self._feature_matrix(test)
        y_train, y_test = train["next_hour_pue"], test["next_hour_pue"]

        self.model = GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, random_state=self.seed,
        )
        self.model.fit(x_train, y_train)

        pred = self.model.predict(x_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        importances = pd.Series(
            self.model.feature_importances_, index=self.feature_columns
        ).sort_values(ascending=False)

        self.report = ModelReport(
            rmse=rmse,
            mae=float(mean_absolute_error(y_test, pred)),
            r2=float(r2_score(y_test, pred)),
            n_train=len(train), n_test=len(test),
            feature_importances=importances,
        )
        return self.report

    # ------------------------------------------------------------------
    # C. Risk classification
    # ------------------------------------------------------------------
    @staticmethod
    def classify_risk(pue_degradation: float) -> str:
        """Map predicted next-hour PUE delta to an interpretable band."""
        low, high = RISK_THRESHOLDS
        if pue_degradation < low:
            return "Low"
        if pue_degradation < high:
            return "Moderate"
        return "High"

    def score_latest_conditions(self) -> pd.DataFrame:
        """Score the most recent hour of every region.

        Returns one row per region with current state, predicted
        next-hour PUE, degradation, and the risk class - the table that
        powers the map, KPI row, and prescriptive optimiser.
        """
        if self.model is None:
            self.train_model()

        latest = (
            self.features.dropna(subset=[
                c for c in self.feature_columns if c in self.features.columns
            ])
            .groupby("region", sort=False).tail(1).copy()
        )
        latest["predicted_next_pue"] = self.model.predict(
            self._feature_matrix(latest)
        )
        latest["predicted_degradation"] = (
            latest["predicted_next_pue"] - latest["pue"]
        )
        latest["risk_class"] = latest["predicted_degradation"].apply(
            self.classify_risk
        )
        cols = [
            "region", "latitude", "longitude", "timestamp",
            "pue", "wue", "it_compute_load_mw", "chiller_power_mw",
            "wet_bulb_temp_c", "watershed_stress_index",
            "grid_carbon_intensity_gco2_kwh",
            "environmental_stress_index",
            "predicted_next_pue", "predicted_degradation", "risk_class",
        ]
        return latest[cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # D. Prescriptive optimisation
    # ------------------------------------------------------------------
    def optimize_workload_distribution(
        self, target_hour: Optional[pd.Timestamp] = None,
        max_shift_fraction: float = 0.20,
    ) -> Dict:
        """Greedy sustainability-aware re-allocation of marginal AI load.

        Method
        ------
        1. Score every region's *placement cost* = blend of environmental
           stress, predicted PUE degradation, and carbon intensity.
        2. Donors: regions in the worst placement-cost tier.
        3. Receivers: regions with placement headroom (low stress, low
           predicted degradation), weighted by how much headroom.
        4. Shift up to `max_shift_fraction` of each donor's current AI
           load, split across receivers proportionally to headroom.
        5. Estimate avoided CO2 (carbon-intensity delta x energy moved,
           including the PUE overhead delta) and avoided water
           (donor evaporative intensity x energy no longer cooled there).

        Deterministic, transparent, and explainable line-by-line.
        """
        scores = self.score_latest_conditions()
        if target_hour is None:
            target_hour = scores["timestamp"].max()

        # Placement cost: higher = worse place to put the next megawatt.
        s = scores.copy()
        s["placement_cost"] = (
            0.5 * s["environmental_stress_index"]
            + 0.3 * np.clip(s["predicted_degradation"], 0, None)
              / max(RISK_THRESHOLDS[1], 1e-9)
            + 0.2 * (s["grid_carbon_intensity_gco2_kwh"]
                     / s["grid_carbon_intensity_gco2_kwh"].max())
        )
        s = s.sort_values("placement_cost", ascending=False)

        donors = s.head(1)            # single worst region donates
        receivers = s.tail(2).copy()  # two best regions absorb

        # Receiver headroom weights (inverse of placement cost).
        inv = 1.0 / (receivers["placement_cost"] + 1e-6)
        receivers["weight"] = inv / inv.sum()

        shifts: List[WorkloadShift] = []
        for _, donor in donors.iterrows():
            shift_total_mw = float(donor["it_compute_load_mw"]) * max_shift_fraction
            # Donor-side intensities for savings estimates.
            donor_water_lph_per_mw = (
                self.raw.loc[self.raw["region"] == donor["region"],
                             "water_consumption_gpm"].tail(24).mean()
                * GPM_TO_LITRES_PER_HOUR
                / max(donor["it_compute_load_mw"], 1e-6)
            )
            for _, recv in receivers.iterrows():
                mw = shift_total_mw * float(recv["weight"])
                energy_mwh = mw * HOURS_PER_SHIFT_WINDOW

                # CO2: energy now consumed at receiver's intensity instead
                # of donor's, with each side's PUE overhead applied.
                # Unit check: MWh x gCO2/kWh = (1000 kWh) x g/kWh / (1000 g/kg)
                # = kg, so no further conversion is needed.
                donor_kg = (energy_mwh * donor["pue"]
                            * donor["grid_carbon_intensity_gco2_kwh"])
                recv_kg = (energy_mwh * recv["predicted_next_pue"]
                           * recv["grid_carbon_intensity_gco2_kwh"])
                co2_saved = max(donor_kg - recv_kg, 0.0)

                # Water: assume receiver adds negligible evaporative load
                # relative to donor's stressed basin (conservative for
                # cool-climate receivers using economiser cooling).
                water_saved = (donor_water_lph_per_mw * mw
                               * HOURS_PER_SHIFT_WINDOW
                               * float(donor["watershed_stress_index"]))

                shifts.append(WorkloadShift(
                    source=donor["region"],
                    destination=recv["region"],
                    shift_mw=round(mw, 1),
                    shift_pct_of_source=round(
                        100.0 * mw / donor["it_compute_load_mw"], 1),
                    co2_saved_kg=round(co2_saved, 0),
                    water_saved_litres=round(water_saved, 0),
                    rationale=(
                        f"{recv['region']} shows lower wet-bulb stress "
                        f"({recv['wet_bulb_temp_c']:.1f}C vs "
                        f"{donor['wet_bulb_temp_c']:.1f}C) and "
                        f"{recv['risk_class'].lower()} predicted PUE "
                        f"degradation risk."
                    ),
                ))

        window_start = pd.Timestamp(target_hour)
        window_end = window_start + pd.Timedelta(hours=HOURS_PER_SHIFT_WINDOW)
        donor0 = donors.iloc[0]
        total_pct = sum(x.shift_pct_of_source for x in shifts)
        dest_names = " and ".join(
            sorted({x.destination for x in shifts})
        )
        headline = (
            f"Reduce {donor0['region']} AI training load by "
            f"{total_pct:.0f}% for {window_start:%H:%M}-{window_end:%H:%M} "
            f"and redirect capacity to {dest_names}, which currently show "
            f"lower wet-bulb stress and lower predicted PUE degradation."
        )

        return {
            "target_window": (window_start, window_end),
            "headline": headline,
            "shifts": shifts,
            "total_co2_saved_kg": round(sum(x.co2_saved_kg for x in shifts), 0),
            "total_water_saved_litres": round(
                sum(x.water_saved_litres for x in shifts), 0),
            "donor_region": donor0["region"],
            "scores": s,
        }


if __name__ == "__main__":
    from data_generator import DataCenterSimulator

    engine = BioDigitalTwinEngine(DataCenterSimulator().generate())
    rep = engine.train_model()
    print(f"RMSE={rep.rmse:.4f}  MAE={rep.mae:.4f}  R2={rep.r2:.3f}  "
          f"(train={rep.n_train}, test={rep.n_test})")
    print("\nTop features:\n", rep.feature_importances.head(8).round(4))
    print("\nLatest scores:\n",
          engine.score_latest_conditions()[
              ["region", "pue", "predicted_next_pue", "risk_class"]
          ])
    plan = engine.optimize_workload_distribution()
    print("\n" + plan["headline"])
    print(f"CO2 saved: {plan['total_co2_saved_kg']:.0f} kg | "
          f"Water saved: {plan['total_water_saved_litres']:.0f} L")
