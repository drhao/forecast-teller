#!/usr/bin/env python
"""Build docs/dengue/ (visual report) from dengue_ewarn outputs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dengue_ewarn import SITE_DIR  # noqa: E402
from dengue_ewarn.site import build  # noqa: E402

d = build()
print(f"site → {SITE_DIR} | figures: {d['cases'] + [d['illustration']['figure']]} | AUC asof_adj tfm {d['auc']['asof_adj']['tfm']}")
