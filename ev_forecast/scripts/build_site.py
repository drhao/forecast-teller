#!/usr/bin/env python
"""Build docs/ev/ (data JSON, narrative, report) from ev_forecast outputs."""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ev_forecast.site import main  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--joint-label", default="聯合三變量（門急診合計 + 門診 + RODS）+ 學校行事曆 + 假日天數")
ap.add_argument("--uni-label", default="單變量 + 學校行事曆 + 假日天數")
a = ap.parse_args()
main({"ev_oe": a.joint_label, "ev_out": a.joint_label, "ev_rods": a.joint_label, "ev_rods_pct": a.uni_label})
