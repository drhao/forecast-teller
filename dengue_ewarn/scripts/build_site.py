#!/usr/bin/env python
"""Build docs/dengue/ (visual report) from dengue_ewarn outputs.

  python scripts/build_site.py                                   # tag dengue（台南 + 高雄）
  python scripts/build_site.py --tag dengue_all --panel dengue_all  # 第三輪：全台 274 鄉鎮
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dengue_ewarn import SITE_DIR  # noqa: E402
from dengue_ewarn.site import build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="dengue"); ap.add_argument("--panel", default=None)
ap.add_argument("--site-dir", default=None, help="override output dir (default docs/dengue)")
args = ap.parse_args()
d = build(tag=args.tag, panel=args.panel, site_dir=args.site_dir)
print(f"site → {args.site_dir or SITE_DIR} | figures: {d['cases'] + [d['illustration']['figure']]} | AUC asof_adj tfm {d['auc']['asof_adj']['tfm']} | scope {d['scope_text']}")
