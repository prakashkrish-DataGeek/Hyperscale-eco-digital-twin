"""
data_generator.py
=================

Synthetic telemetry simulator for the Hyperscale Bio-Digital Twin.

Generates a deterministic, physically plausible 30-day, 1-hour resolution
operational-climate time series for four strategic AI data centre regions:

    1. Northern Virginia (Loudoun County)  - humid subtropical, grid congestion
    2. West Texas                          - extreme dry heat, water scarcity
    3. Dublin, Ireland                     - cool marine, sustainability pressure
    4. Frankfurt, Germany                  - continental, episodic heat waves

Engineering rationale
---------------------
The simulation is built around three coupled subsystems:

* CLIMATE: diurnal and weekly sinusoids plus region-specific noise drive
  ambient temperature and humidity. Wet-bulb temperature is derived from
  these using the Stull (2011) empirical approximation, because wet-bulb
  is the binding physical constraint on evaporative and economiser cooling.

* COMPUTE: IT load follows a business-hours weekly rhythm with
  superimposed AI training "surge windows" (large batch jobs scheduled in
  specific hours), reflecting how LLM training campaigns actually load
  hyperscale campuses.

* THERMAL PLANT: chiller power responds to BOTH compute heat rejection and
  environmental stress, with a one-hour exponential lag term to capture
  thermal inertia of chilled-water loops and building mass. PUE and WUE
  are then derived quantities, never sampled independently - this keeps
  the physics internally consistent.

All randomness flows from a single seeded NumPy Generator so every run of
the application reproduces the same world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

# Single source of truth for reproducibility across the whole platform.
GLOBAL_SEED: int = 42

# Simulation horizon: 30 days at 1-hour resolution.
SIMULATION_DAYS: int = 30
HOURS: int = SIMULATION_DAYS * 24


@dataclass(frozen=True)
class RegionProfile:
    """Static physical and infrastructural description of one region.

    Attributes capture the *baseline* behaviour of a region; dynamic
    behaviour (cycles, surges, lag) is layered on in the simulator.
    """

    name: str
    latitude: float
    longitude: float
    base_temp_c: float            # mean ambient temperature
    diurnal_amplitude_c: float    # half peak-to-trough daily swing
    base_humidity_pct: float      # mean relative humidity
    humidity_amplitude_pct: float
    watershed_stress_base: float  # 0..1, chronic basin-level water stress
    carbon_base: float            # gCO2/kWh grid average
    carbon_solar_dip: float       # midday carbon reduction (solar-rich grids)
    it_load_base_mw: float        # mean IT load
    it_load_surge_mw: float       # extra MW during AI training surges
    surge_hours: tuple            # UTC hours when training campaigns run
    cooling_efficiency: float     # plant quality: lower = better chillers
    temp_noise_c: float           # short-term weather volatility
    description: str = field(default="")


def _region_profiles() -> List[RegionProfile]:
    """Region catalogue.

    Values are stylised but anchored in public domain knowledge:
    - Loudoun County summers are hot and humid with a congested PJM grid.
    - West Texas (ERCOT West) sees >38C dry heat, deep solar midday dips
      in carbon intensity, and severe Ogallala/Permian water stress.
    - Dublin rarely exceeds 23C; SEAI grid is moderately carbon intensive
      but planning pressure on data centre sustainability is high.
    - Frankfurt mixes a dense industrial grid with green-power procurement
      limits and increasingly frequent continental heat waves.
    """
    return [
        RegionProfile(
            name="Northern Virginia",
            latitude=39.05, longitude=-77.49,
            base_temp_c=26.0, diurnal_amplitude_c=5.5,
            base_humidity_pct=68.0, humidity_amplitude_pct=12.0,
            watershed_stress_base=0.45,
            carbon_base=380.0, carbon_solar_dip=40.0,
            it_load_base_mw=92.0, it_load_surge_mw=26.0,
            surge_hours=(13, 14, 15, 16, 17),
            cooling_efficiency=0.92, temp_noise_c=1.6,
            description="High humidity, grid congestion, heavy AI utilisation",
        ),
        RegionProfile(
            name="West Texas",
            latitude=31.99, longitude=-102.08,
            base_temp_c=31.0, diurnal_amplitude_c=9.0,
            base_humidity_pct=32.0, humidity_amplitude_pct=10.0,
            watershed_stress_base=0.82,
            carbon_base=410.0, carbon_solar_dip=140.0,
            it_load_base_mw=74.0, it_load_surge_mw=34.0,
            surge_hours=(9, 10, 11, 12, 13),
            cooling_efficiency=1.00, temp_noise_c=2.4,
            description="Extreme dry heat, solar-rich grid, severe water scarcity",
        ),
        RegionProfile(
            name="Dublin",
            latitude=53.35, longitude=-6.26,
            base_temp_c=14.5, diurnal_amplitude_c=3.5,
            base_humidity_pct=80.0, humidity_amplitude_pct=8.0,
            watershed_stress_base=0.18,
            carbon_base=290.0, carbon_solar_dip=25.0,
            it_load_base_mw=58.0, it_load_surge_mw=16.0,
            surge_hours=(20, 21, 22, 23),
            cooling_efficiency=0.78, temp_noise_c=1.2,
            description="Cool marine climate, strong sustainability pressure",
        ),
        RegionProfile(
            name="Frankfurt",
            latitude=50.11, longitude=8.68,
            base_temp_c=20.0, diurnal_amplitude_c=6.0,
            base_humidity_pct=62.0, humidity_amplitude_pct=10.0,
            watershed_stress_base=0.35,
            carbon_base=340.0, carbon_solar_dip=70.0,
            it_load_base_mw=66.0, it_load_surge_mw=20.0,
            surge_hours=(0, 1, 2, 3, 22, 23),
            cooling_efficiency=0.86, temp_noise_c=1.8,
            description="Industrial grid constraints, episodic heat waves",
        ),
    ]


class DataCenterSimulator:
    """Deterministic generator of coupled climate-operations telemetry.

    Usage
    -----
    >>> sim = DataCenterSimulator()
    >>> df = sim.generate()
    """

    def __init__(self, seed: int = GLOBAL_SEED, hours: int = HOURS) -> None:
        self.seed = int(seed)
        self.hours = int(hours)
        self.profiles: List[RegionProfile] = _region_profiles()
        self._rng = np.random.default_rng(self.seed)
        # Anchor the series so 'latest hour' is stable and reproducible.
        self.end_time = pd.Timestamp("2026-06-12 12:00:00")
        self.timestamps = pd.date_range(
            end=self.end_time, periods=self.hours, freq="h"
        )

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------
    @staticmethod
    def wet_bulb_stull(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
        """Wet-bulb temperature via the Stull (2011) approximation.

        Valid for RH 5-99% and T -20..50C, which comfortably covers the
        simulated envelope. Wet-bulb governs how much 'free' or
        evaporative cooling a facility can extract from outside air,
        so it is the single most important climate driver of PUE.
        """
        t = np.asarray(temp_c, dtype=float)
        rh = np.clip(np.asarray(rh_pct, dtype=float), 5.0, 99.0)
        tw = (
            t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
            + np.arctan(t + rh)
            - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
            - 4.686035
        )
        return tw

    @staticmethod
    def _lagged_response(driver: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """First-order exponential smoothing - models thermal inertia.

        Chilled-water loops, thermal storage, and building mass mean the
        plant responds to stress with a delay; alpha=0.45 gives roughly a
        1-2 hour effective lag, which matches large-campus behaviour.
        """
        out = np.empty_like(driver, dtype=float)
        out[0] = driver[0]
        for i in range(1, len(driver)):
            out[i] = alpha * driver[i] + (1.0 - alpha) * out[i - 1]
        return out

    # ------------------------------------------------------------------
    # Per-subsystem builders (one region at a time)
    # ------------------------------------------------------------------
    def _climate_series(self, p: RegionProfile) -> Dict[str, np.ndarray]:
        """Ambient temperature, humidity and wet-bulb for one region."""
        n = self.hours
        hour_of_day = self.timestamps.hour.to_numpy(dtype=float)
        day_index = np.arange(n) / 24.0

        # Diurnal cycle peaking ~15:00 local-ish; we keep everything in a
        # single clock for simplicity - relative dynamics matter more than
        # absolute time zones for this prototype.
        diurnal = np.sin((hour_of_day - 9.0) / 24.0 * 2.0 * np.pi)

        # Slow synoptic swing (multi-day weather systems) + heat-wave bumps.
        synoptic = 1.8 * np.sin(day_index / 6.5 * 2.0 * np.pi + p.latitude)
        heat_wave = np.zeros(n)
        if p.name in ("Frankfurt", "West Texas"):
            # Two scripted heat-wave episodes for episodic-stress regions.
            for start_day in (8, 21):
                s, e = start_day * 24, (start_day + 3) * 24
                heat_wave[s:e] += 4.5 * np.hanning(e - s)

        noise = self._rng.normal(0.0, p.temp_noise_c, n)
        temp = (
            p.base_temp_c
            + p.diurnal_amplitude_c * diurnal
            + synoptic
            + heat_wave
            + noise
        )

        # Humidity moves inversely with temperature within the day
        # (warm afternoons are relatively drier), plus its own noise.
        rh = (
            p.base_humidity_pct
            - p.humidity_amplitude_pct * diurnal
            + self._rng.normal(0.0, 3.0, n)
        )
        rh = np.clip(rh, 8.0, 99.0)

        wet_bulb = self.wet_bulb_stull(temp, rh)
        return {"ambient_temp_c": temp, "relative_humidity_pct": rh,
                "wet_bulb_temp_c": wet_bulb}

    def _grid_series(self, p: RegionProfile) -> Dict[str, np.ndarray]:
        """Grid carbon intensity and watershed stress for one region."""
        n = self.hours
        hour_of_day = self.timestamps.hour.to_numpy(dtype=float)

        # Solar production carves a midday dip in carbon intensity;
        # depth of the dip is region-specific (deep in West Texas).
        solar_shape = np.clip(np.sin((hour_of_day - 6.0) / 12.0 * np.pi), 0.0, None)
        carbon = (
            p.carbon_base
            - p.carbon_solar_dip * solar_shape
            + self._rng.normal(0.0, 12.0, n)
        )
        carbon = np.clip(carbon, 60.0, 700.0)

        # Watershed stress is slow-moving: chronic base + gentle drift that
        # worsens during hot spells (evaporation, competing demand).
        day_index = np.arange(n) / 24.0
        drift = 0.05 * np.sin(day_index / 15.0 * np.pi)
        stress = p.watershed_stress_base + drift + self._rng.normal(0.0, 0.012, n)
        stress = np.clip(stress, 0.02, 0.98)
        return {"grid_carbon_intensity_gco2_kwh": carbon,
                "watershed_stress_index": stress}

    def _compute_series(self, p: RegionProfile) -> np.ndarray:
        """IT compute load with weekly rhythm and AI training surges."""
        n = self.hours
        hour_of_day = self.timestamps.hour.to_numpy(dtype=float)
        weekday = self.timestamps.weekday.to_numpy(dtype=float)

        # Enterprise/inference base load: higher on weekdays, daytime.
        weekly = np.where(weekday < 5, 1.0, 0.86)
        daily = 1.0 + 0.10 * np.sin((hour_of_day - 8.0) / 24.0 * 2.0 * np.pi)

        load = p.it_load_base_mw * weekly * daily

        # AI training campaigns: scheduled surge windows with ramp-in/out,
        # plus occasional multi-day campaigns (deterministic via seed).
        in_surge = np.isin(self.timestamps.hour, p.surge_hours).astype(float)
        campaign = (self._rng.random(SIMULATION_DAYS) < 0.55).astype(float)
        campaign_by_hour = np.repeat(campaign, 24)[:n]
        load += p.it_load_surge_mw * in_surge * campaign_by_hour

        load += self._rng.normal(0.0, 1.5, n)
        return np.clip(load, 5.0, None)

    def _thermal_plant(
        self, p: RegionProfile, climate: Dict[str, np.ndarray],
        it_load: np.ndarray, stress: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Chiller power, water use, PUE and WUE - derived, not sampled.

        Chiller demand model:
            chiller_mw = f(IT heat rejection) * g(wet-bulb difficulty)
        then passed through a thermal-inertia lag. PUE follows directly:
            PUE = (IT + chiller + fixed overhead) / IT
        """
        wb = climate["wet_bulb_temp_c"]

        # Cooling difficulty rises non-linearly once wet-bulb passes ~18C:
        # economiser hours vanish and mechanical chillers carry full load.
        difficulty = 1.0 + 0.035 * np.clip(wb - 10.0, 0.0, None) ** 1.25 / 10.0

        base_ratio = 0.16 * p.cooling_efficiency  # MW chiller per MW IT, mild day
        instantaneous = it_load * base_ratio * difficulty
        chiller = self._lagged_response(instantaneous, alpha=0.45)
        chiller += self._rng.normal(0.0, 0.4, self.hours)
        chiller = np.clip(chiller, 0.5, None)

        # Fixed electrical/mechanical overhead (UPS losses, lighting, fans).
        overhead = 0.045 * it_load + 2.0

        pue = (it_load + chiller + overhead) / it_load
        pue = np.clip(pue, 1.04, 2.2)

        # Water: evaporative make-up scales with chiller load and gets
        # WORSE when the basin is stressed (warmer source water, higher
        # cycles of concentration limits). Dry-cooled plants (low base
        # humidity regions often use hybrid) still consume some water.
        gpm_per_mw = 11.0 + 9.0 * stress
        water_gpm = chiller * gpm_per_mw + self._rng.normal(0.0, 3.0, self.hours)
        water_gpm = np.clip(water_gpm, 0.0, None)

        # WUE in L/kWh of IT energy: convert gal/min -> L/hr over MW*1h.
        litres_per_hour = water_gpm * 3.785 * 60.0
        wue = litres_per_hour / (it_load * 1000.0)
        wue = np.clip(wue, 0.02, 4.0)

        return {"chiller_power_mw": chiller, "pue": pue,
                "water_consumption_gpm": water_gpm, "wue": wue}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_region(self, p: RegionProfile) -> pd.DataFrame:
        """Full coupled series for a single region."""
        climate = self._climate_series(p)
        grid = self._grid_series(p)
        it_load = self._compute_series(p)
        plant = self._thermal_plant(
            p, climate, it_load, grid["watershed_stress_index"]
        )

        df = pd.DataFrame({
            "timestamp": self.timestamps,
            "region": p.name,
            "latitude": p.latitude,
            "longitude": p.longitude,
            "ambient_temp_c": climate["ambient_temp_c"],
            "relative_humidity_pct": climate["relative_humidity_pct"],
            "wet_bulb_temp_c": climate["wet_bulb_temp_c"],
            "watershed_stress_index": grid["watershed_stress_index"],
            "grid_carbon_intensity_gco2_kwh":
                grid["grid_carbon_intensity_gco2_kwh"],
            "it_compute_load_mw": it_load,
            "chiller_power_mw": plant["chiller_power_mw"],
            "water_consumption_gpm": plant["water_consumption_gpm"],
            "pue": plant["pue"],
            "wue": plant["wue"],
        })
        return df

    def generate(self) -> pd.DataFrame:
        """Unified multi-region DataFrame, sorted by region then time.

        Returns
        -------
        pd.DataFrame
            One row per region-hour; ~2,880 rows for 30 days x 4 regions.
        """
        frames = [self.generate_region(p) for p in self.profiles]
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values(["region", "timestamp"]).reset_index(drop=True)

        # Defensive sanity assertions - fail loudly if physics breaks.
        assert df["pue"].between(1.0, 2.5).all(), "PUE out of plausible range"
        assert df["wet_bulb_temp_c"].le(df["ambient_temp_c"] + 0.5).all(), (
            "Wet-bulb exceeded dry-bulb - check humidity inputs"
        )
        assert not df.isna().any().any(), "NaNs in generated telemetry"
        return df


if __name__ == "__main__":
    simulator = DataCenterSimulator()
    data = simulator.generate()
    print(data.groupby("region")[["pue", "wue", "wet_bulb_temp_c"]]
          .mean().round(3))
    print(f"\nRows: {len(data)}, span: {data.timestamp.min()} "
          f"-> {data.timestamp.max()}")
