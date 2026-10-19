"""Shared pieces for the generated pages of the GitHub Pages site (docs/).

The navigation itself lives in docs/assets/nav.js (single source of truth); generated pages only emit an
empty <header class="masthead" data-disease=… data-page=…> followed by the nav.js tag. Method pages for all
four diseases use `method_page` so their layout stays identical.
"""
from __future__ import annotations

ASSET_V = "nav2"   # bump to bust browser caches of epi.css / charts.js / nav.js
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:'
         'wght@400;500;600;700&family=Noto+Serif+TC:wght@600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">')
DISEASE_NAME = {"flu": "類流感", "covid": "新冠", "ev": "腸病毒", "dengue": "登革熱"}


def masthead(disease: str, page: str, up: str = "../") -> str:
    return (f'<header class="masthead" data-disease="{disease}" data-page="{page}"></header>\n'
            f'<script src="{up}assets/nav.js?v={ASSET_V}"></script>')


def page_head(title: str, disease: str, page: str, up: str = "../", charts: bool = True, extra_head: str = "") -> str:
    """<!doctype> … <body> + masthead. `up` is the relative path from the page to docs/."""
    scripts = (f'<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>\n'
               f'<script src="{up}assets/charts.js?v={ASSET_V}" defer></script>') if charts else ""
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
{FONTS}
<link rel="stylesheet" href="{up}assets/epi.css?v={ASSET_V}">
{scripts}{extra_head}
</head><body>
{masthead(disease, page, up)}
"""


def reading_items(threshold_line: str) -> list[str]:
    """How to read a weekly quantile forecast; shared by the three weekly-forecast diseases."""
    return [
        "<b>先看區間，不看單點。</b>模型對每個未來週輸出 9 個分位數（q10–q90）。中位數是「一半機會比它高、一半比它低」；q10–q90 為 80% 預測區間，q20–q80 為 60%。區間寬代表不確定，是資訊不是缺陷。",
        "<b>預測起點是最新的完整申報週。</b>本週尚未結束、或申報明顯未齊的週不進入模型；各指標的起點可能不同（例如重症依發病週，通報較慢）。",
        "<b>「下週續升機率」</b>是第 1 週分布高於起點週觀測值的比例，由 9 個分位數線性內插而得，上下限 5%–95%。",
        threshold_line,
        "<b>「已驗證」</b>標記表示該目標週的實際值已公布，可直接比對；每週的預測都存檔（forecast_log），並與日後的實際值比對評估。",
        "<b>1–2 週最可靠，3–4 週僅供規劃參考。</b>回測中誤差隨預測週數大致線性上升。",
    ]


THRESHOLD_PROB_LINE = ("<b>「未來 4 週內超過閾值的機率」</b>：模型只給每週各自的分布，沒有聯合路徑。主值取 4 週中最高的單週機率，等於假設各週誤差同向"
                       "（同一模型連續週的誤差高度正相關，這是合理近似）；另列各週獨立時的上界 1 − Π(1 − p<sub>h</sub>)。"
                       "指標已在閾值之上時，改報「4 週內降到閾值以下」的機率。")


def method_page(disease: str, title: str, kicker: str, h1: str, lead: str, sections: list[tuple[str, str, str]], footer: str,
                up: str = "../") -> str:
    """sections: [(heading, english, html_body)]; numbered 01, 02, … ."""
    body = "".join(f"""
  <section>
    <div class="section-head"><span class="section-num">{i:02d}</span><h2>{h}</h2><span class="en">{en}</span></div>
    {html}
  </section>""" for i, (h, en, html) in enumerate(sections, 1))
    return page_head(title, disease, "method", up, charts=False) + f"""<main>
  <div class="hero-kicker">{kicker}</div>
  <h1 class="page-title">{h1}</h1>
  <p class="lead">{lead}</p>{body}
</main>
<footer class="site-footer">{footer}</footer>
</body></html>"""


def table(head: list[str], rows: list[list[str]], num_cols: tuple[int, ...] = ()) -> str:
    nc = ' class="num"'
    th = "".join(f"<th{nc if i in num_cols else ''}>{h}</th>" for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f"<td{nc if i in num_cols else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tbl-wrap"><table class="tbl compact"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def ul(items: list[str], cls: str = "findings") -> str:
    return f'<ul class="{cls}">' + "".join(f"<li>{x}</li>" for x in items) + "</ul>"
