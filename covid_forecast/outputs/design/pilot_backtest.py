"""COVID pilot backtest (planning only): national weekly series, origins 2024w01–2026w36, h=1–4."""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from forecast_teller.backtest import Config, run_config, summarize, SEGMENTS
from forecast_teller.covariates import future_covariates
S = "/private/tmp/claude-502/-Users-drhao-Documents-GitHub-forecast-teller/d01a93a5-3949-4980-82fd-4d6358394909/scratchpad"
p = pd.read_csv(f"{S}/covid_national_weekly_probe.csv", index_col=0, dtype={0: str}); p.index = p.index.astype(str)
nat = pd.DataFrame({"covid_oe": p.nidss_oe, "covid_out": p.open_out, "covid_rods": p.rods_er, "covid_inp": p.open_inp,
                    "covid_er_est": p.nhi_er_est}).loc["202127":"202639"]
nat["covid_rods"] = nat["covid_rods"].fillna(0.0)
SEGMENTS.clear(); SEGMENTS.update({"y2024": (2024, 2024), "y2025": (2025, 2025), "y2026": (2026, 2026), "all": (2024, 2026)})
END, ORG = "202639", "202401"
def cfgs(start):
    base = dict(target="covid_oe", start=start, end=END, origin_start=ORG, horizon=4)
    return [Config("naive", model="naive", **base), Config("ma3", model="ma3", **base), Config("snaive", model="snaive", **base),
            Config("ets", model="ets", **base), Config("theta", model="theta", **base),
            Config("tfm_uni", **base), Config("tfm_uni_log1p", log1p=True, **base),
            Config("tfm_uni_hol", covariates=("cny", "holiday"), **base),
            Config("tfm_uni_season_hol", covariates=("season", "cny", "holiday"), **base),
            Config("tfm_mv_out_rods", targets=("covid_oe", "covid_out", "covid_rods"), **base),
            Config("tfm_mv_out_rods_hol", targets=("covid_oe", "covid_out", "covid_rods"), covariates=("cny", "holiday"), **base),
            Config("tfm_mv_out_rods_inp_hol", targets=("covid_oe", "covid_out", "covid_rods", "covid_inp"), covariates=("cny", "holiday"), **base)]
from forecast_teller.model_timesfm3 import TimesFM3Model
model = TimesFM3Model(backend="mlx", batch_size=8)
res = []
for start, suffix in [("202127", ""), ("202312", "_ctx2023")]:
    for cfg in cfgs(start):
        if suffix and cfg.model != "timesfm3": continue
        cfg.name = cfg.name + suffix
        t0 = time.time()
        r = run_config(cfg, model=model if cfg.model == "timesfm3" else None, panel=nat, covariate_fn=future_covariates, verbose=False)
        res.append(r); print(f"{cfg.name:28s} {len(r):5d} rows {time.time()-t0:5.1f}s", flush=True)
df = pd.concat(res, ignore_index=True); df.to_parquet(f"{S}/covid_pilot_forecasts.parquet")
summ = summarize(df, thresholds={"covid_oe": 10000.0}); summ.to_csv(f"{S}/covid_pilot_summary.csv", index=False)
# pivot: config x h for segment all, WIS relative to naive
a = summ[(summ.target == "covid_oe") & (summ.segment == "all")]
piv = a.pivot_table(index="config", columns="h", values="wis")
naive = piv.loc["naive"]
rel = (piv / naive).round(2); rel["mean"] = (piv.mean(axis=1) / naive.mean()).round(2)
print("\n== WIS relative to naive (all origins 2024w01–2026w36), by h ==\n", rel.sort_values("mean").to_string())
extra = a.groupby("config")[["mape", "cov80", "dir_hit"]].mean().round(2) if "cov80" in a.columns else a.groupby("config")[["mape"]].mean()
print("\n== mean over h: MAPE / 80% coverage / direction hit ==\n", extra.loc[rel.sort_values("mean").index].to_string())
print("\n== n origins ==", a.groupby("config")["n"].first().to_dict() if "n" in a.columns else "")
print("\ncolumns:", list(summ.columns))
for seg in ["y2024", "y2025", "y2026"]:
    b = summ[(summ.target == "covid_oe") & (summ.segment == seg)].pivot_table(index="config", columns="h", values="wis")
    print(f"\n== {seg}: WIS/naive mean over h ==\n", (b.mean(axis=1) / b.loc["naive"].mean()).round(2).sort_values().to_string())
