#!/usr/bin/env python
"""Build docs/covid/ (data JSON, narrative, report) from covid_forecast outputs."""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_forecast.site import main  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--joint-label", default="聯合二變量（門診 + RODS）+ 春節 + 假日天數")
ap.add_argument("--uni-label", default="單變量 + 春節 + 假日天數（context 自 2021w27）")
a = ap.parse_args()
main({"covid_oe": a.uni_label, "covid_out": a.joint_label, "covid_rods": a.joint_label,
      "covid_rods_pct": a.uni_label, "covid_inp": a.uni_label, "covid_oe_alt": a.uni_label})
