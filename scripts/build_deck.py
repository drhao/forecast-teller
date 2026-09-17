#!/usr/bin/env python
"""Build the 6-slide results deck (python-pptx), styled with the epidemic dataviz palette."""
import json, sys, datetime as dt
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt, Emu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast_teller import OUTPUT_DIR, ROOT  # noqa: E402

C = {"p50": "F6F9F6", "p100": "E8EEE7", "p300": "B4C9B1", "p500": "739A6D", "p600": "5D7F58", "p800": "374C34", "p900": "253423",
     "n0": "FFFFFF", "n100": "F2F3F1", "n200": "E4E7E4", "n300": "CACFC9", "n400": "A2ABA0", "n500": "7A8778", "n600": "5D675B", "n700": "444C43", "n800": "2C312B", "n900": "181B18",
     "slate": "587A9D", "mustard": "A8821F", "alert": "BE373C"}
FONT = "PingFang TC"
SL = OUTPUT_DIR / "slides"
bt = json.loads((ROOT / "docs/data/backtest.json").read_text(encoding="utf-8"))
latest = json.loads((ROOT / "docs/data/latest.json").read_text(encoding="utf-8"))
T = bt["targets"]["nhi_out_ili"]; POST = T["segments"]["post_covid"]; LB = {x["config"]: x for x in POST["leaderboard"]}
ER = bt["targets"]["nhi_er_ili"]; ERB = {x["config"]: x for x in ER["segments"]["post_covid"]["leaderboard"]}[ER["best"]]
RD = bt["targets"]["rods_ili_pct"]; RDB = {x["config"]: x for x in RD["segments"]["post_covid"]["leaderboard"]}[RD["best"]]
CTY = {x["config"]: x for x in bt["groups"]["county"]["segments"]["post_covid"]}; AGE = {x["config"]: x for x in bt["groups"]["age"]["segments"]["post_covid"]}
best, naive = LB[T["best"]], LB["naive"]
impr = [round(100 * (1 - a / b)) for a, b in zip(POST["wis_by_h"][T["best"]], POST["wis_by_h"]["naive"])]
NAR = latest["narrative"]; OUT_IND = next(i for i in latest["indicators"] if i["key"] == "nhi_out_ili")
TODAY = dt.date.today().isoformat()

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def rgb(h): return RGBColor.from_string(h)


def _ea(run):
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = OxmlElement(tag); rPr.append(el)
        el.set("typeface", FONT)


def text(slide, x, y, w, h, paras, size=14, color="n800", bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, bullets=False, space_after=4, line=1.15, font=FONT):
    """paras: str | list of str | list of list-of-runs; a run is str or (str, {size,bold,color})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.word_wrap = True; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.vertical_anchor = anchor
    if isinstance(paras, str): paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space_after); p.line_spacing = line
        runs = para if isinstance(para, list) else [para]
        for r in runs:
            t, o = (r, {}) if isinstance(r, str) else r
            run = p.add_run(); run.text = t; f = run.font
            f.name = o.get("font", font); f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold); f.color.rgb = rgb(C[o.get("color", color)]); _ea(run)
        if bullets:
            pPr = p._p.get_or_add_pPr(); pPr.set("marL", str(Emu(Inches(0.22)))); pPr.set("indent", str(-Emu(Inches(0.22))))
            bu = OxmlElement("a:buChar"); bu.set("char", "•"); pPr.append(bu)
    return tb


def rect(slide, x, y, w, h, fill="n100", line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h)); s.fill.solid(); s.fill.fore_color.rgb = rgb(C[fill])
    if line: s.line.color.rgb = rgb(C[line]); s.line.width = Pt(0.75)
    else: s.line.fill.background()
    s.shadow.inherit = False; return s


def circle_num(slide, x, y, n, d=0.36):
    c = rect(slide, x, y, d, d, fill="p500", shape=MSO_SHAPE.OVAL); tf = c.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = str(n); r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = rgb(C["n0"]); r.font.name = FONT; _ea(r)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE


def title(slide, t, sub=None, kicker=None):
    if kicker: text(slide, 0.6, 0.38, 8, 0.25, kicker, size=10, color="p600", bold=True)
    text(slide, 0.6, 0.6, 12.1, 0.6, t, size=26, color="p900", bold=True)
    if sub: text(slide, 0.6, 1.22, 12.1, 0.35, sub, size=12.5, color="n600")


def footer(slide, n):
    text(slide, 0.6, 7.05, 9, 0.25, "forecast-teller · TimesFM 3.0 台灣流感預測 · 資料：疾管署資料開放平台 · 圖表依《疫情資料視覺化指引》", size=8.5, color="n500")
    text(slide, 12.2, 7.05, 0.5, 0.25, str(n), size=9, color="n500", align=PP_ALIGN.RIGHT)


def table(slide, x, y, w, rows, col_w, header_fill="n100", font_size=10.5, best_row=None, row_h=0.3, num_from=None):
    """num_from: index of the first numeric (right-aligned) column; None = all columns left-aligned."""
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows))); tbl = shp.table
    tblPr = tbl._tbl.tblPr; tblPr.set("firstRow", "0"); tblPr.set("bandRow", "0")
    style = tblPr.find(qn("a:tableStyleId"))
    if style is not None: tblPr.remove(style)
    for j, cw in enumerate(col_w): tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j); cell.margin_left = cell.margin_right = Inches(0.06); cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.fill.solid(); cell.fill.fore_color.rgb = rgb(C[header_fill if i == 0 else ("p50" if i == best_row else "n0")])
            tf = cell.text_frame; tf.word_wrap = True; p = tf.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT if (num_from is not None and j >= num_from) else PP_ALIGN.LEFT
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            r = p.add_run(); r.text = str(val); r.font.size = Pt(font_size); r.font.name = FONT; _ea(r)
            r.font.bold = (i == 0) or (i == best_row and j == 0); r.font.color.rgb = rgb(C["n600" if i == 0 else "n800"])
            tcPr = cell._tc.get_or_add_tcPr()
            for k, side in enumerate(("a:lnL", "a:lnR", "a:lnT", "a:lnB")):  # must precede the fill in a:tcPr
                ln = OxmlElement(side); ln.set("w", "6350" if side == "a:lnB" else "0"); ln.set("cap", "flat"); ln.set("cmpd", "sng"); ln.set("algn", "ctr")
                if side == "a:lnB":
                    sf = OxmlElement("a:solidFill"); clr = OxmlElement("a:srgbClr"); clr.set("val", C["n200"]); sf.append(clr); ln.append(sf)
                else:
                    ln.append(OxmlElement("a:noFill"))
                tcPr.insert(k, ln)
    return tbl


fmt0 = lambda v: f"{v:,.0f}"

# ---------------------------------------------------------------- slide 1 · title (dark)
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, 13.333, 7.5, fill="n900")
text(s, 0.8, 1.15, 11, 0.3, "FORECAST-TELLER · 成果報告 · " + TODAY, size=11, color="p300", bold=True)
text(s, 0.8, 1.6, 11.5, 1.6, "以 TimesFM 3.0 零樣本預測台灣流感疫情", size=40, color="n0", bold=True)
text(s, 0.8, 2.75, 11.5, 0.9, "資料管線、滾動回測、線上 Dashboard：從疾管署監測資料到每週 1–4 週預測", size=18, color="n300")
stats = [(f"{T['n_origins']}", "個回測起點（2018–2025，每週一個）"), (f"−{round(100 * (1 - best['rel_naive']))}%", "最佳設定 WIS 相對 naive（COVID 後）"),
         (f"{best['cov80']:.2f}", "80% 預測區間實際涵蓋率（理想 0.80）"), (f"{best['dir_hit']:.2f}", "方向命中率（naive 退化為 " + f"{naive['dir_hit']:.2f}" + "）")]
for i, (v, lab) in enumerate(stats):
    x = 0.8 + i * 3.05
    text(s, x, 4.2, 2.9, 0.8, v, size=38, color="p300", bold=True)
    text(s, x, 5.05, 2.8, 0.7, lab, size=11, color="n300")
text(s, 0.8, 6.5, 11.5, 0.3, "Google TimesFM 3.0（330M，零樣本、未微調）· 健保、RODS、NIDDS、LARS 實驗室自動通報週資料 · drhao.github.io/forecast-teller", size=11, color="n400")
s.notes_slide.notes_text_frame.text = "開場：專案目標是用時間序列基礎模型做零樣本流感預測，不訓練模型，靠回測驗證可信度，並把結果做成每週更新的線上 dashboard。"

# ---------------------------------------------------------------- slide 2 · data & method
s = prs.slides.add_slide(BLANK); title(s, "資料與方法", "五個疾管署監測來源對齊成週資料面板，TimesFM 3.0 零樣本預測，逐層加入共變數與多變量聯合", kicker="01 · DATA & METHOD")
rect(s, 0.6, 1.75, 6.3, 4.05, fill="n100")
text(s, 0.85, 1.9, 5.8, 0.3, "資料來源（疫情週，Big5 CSV → 自動偵測不完整週）", size=12.5, color="p900", bold=True)
rows = [["來源", "內容", "期間"],
        ["健保申報 門診/住院", "類流感（48X）、流感（487）就診人次、總人次", "2005–2026w36"],
        ["健保申報 急診", "同上，急診", "2016–2026w36"],
        ["RODS 即時疫情監視", "急診類流感人次、百分比（234 家醫院）", "2009–2026w36"],
        ["NIDDS 法定傳染病", "流感併發重症確定病例（發病週）", "2003–2026w33"],
        ["LARS 實驗室自動通報", "A/B 型陽性數、檢驗件數、陽性率", "2015–2026w36"]]
table(s, 0.85, 2.3, 5.85, rows, [1.65, 3.0, 1.2], font_size=9.5, row_h=0.42, num_from=None)
text(s, 0.85, 5.0, 5.8, 0.7, ["22 縣市 × 19 年齡層皆有分解；假日檔（春節、平日假日天數）延伸至 2026 年底；頭尾不完整週與 NIDDS 隱性零值自動處理。"], size=10, color="n700")
steps = [("TimesFM 3.0", "Google 時間序列基礎模型，330M 參數，零樣本、不微調；一次輸出 9 個分位數（q10–q90）。"),
         ("已知未來共變數", "春節週旗標、每週平日假日天數、季節相位；假日天數是最有用的變數。"),
         ("多變量聯合", "門診 + 急診 + RODS + 重症四個序列一起預測（變數注意力），22 縣市 / 18 年齡層也各自聯合。"),
         ("基準模型", "last-value naive、MA3（前 3 週平均）、季節性 naive、AutoETS、Theta。"),
         ("回測設計", "資料窗 2016–2025（522 週），暖機 2 季，2018w01 起每週一個起點共 415 個，預測 1–4 週；分 COVID 前 / 期 / 後三段。")]
for i, (h, d) in enumerate(steps):
    y = 1.8 + i * 0.82
    circle_num(s, 7.3, y + 0.02, i + 1)
    text(s, 7.8, y - 0.02, 4.9, 0.3, h, size=13, color="p900", bold=True)
    text(s, 7.8, y + 0.3, 4.9, 0.55, d, size=10.5, color="n700", line=1.1)
rect(s, 0.6, 6.0, 12.1, 0.85, fill="p50")
text(s, 0.85, 6.1, 11.7, 0.65, [[("評估指標　", {"bold": True, "color": "p900"}), ("WIS（加權區間分數，主指標）、MAE、MAPE、MASE、80%/60% 區間涵蓋率、方向命中率、±10% 容忍帶命中率、閾值命中率與誤報率（門診以第 75 百分位、RODS 以流行閾值 10%）。", {})],
                                 [("決策規則　", {"bold": True, "color": "p900"}), ("以 COVID 後（2023–2025）1–4 週平均 WIS 選定設定，並確認 COVID 前不退步。", {})]], size=10.5, color="n700", space_after=2)
footer(s, 2)
s.notes_slide.notes_text_frame.text = "強調：模型完全零樣本；所有比較都在同一組 415 個起點上；COVID 期（2020–2022）不做決策依據但保留在 context。"

# ---------------------------------------------------------------- slide 3 · backtest results
s = prs.slides.add_slide(BLANK); title(s, "回測結果：全國類流感門診人次（COVID 後 2023–2025，h = 1–4）", f"最佳設定為四變量聯合 + 春節旗標 + 假日天數：WIS 較 last-value naive 低 {round(100*(1-best['rel_naive']))}%，1–4 週分別 −{impr[0]}% / −{impr[1]}% / −{impr[2]}% / −{impr[3]}%", kicker="02 · BACKTEST")
s.shapes.add_picture(str(SL / "wis_by_horizon.png"), Inches(0.6), Inches(1.75), width=Inches(6.4))
lbrows = [["設定", "WIS", "相對 naive", "MAPE %", "80% 涵蓋", "方向命中"]]
for cfg, name in [(T["best"], "四變量聯合＋春節＋假日"), ("tfm_cov_cny_hol", "單變量＋春節＋假日"), ("tfm_expanding", "TimesFM zero-shot"), ("ets", "AutoETS"), ("naive", "last-value naive"), ("ma3", "MA3"), ("snaive", "季節性 naive")]:
    r = LB[cfg]; lbrows.append([name, fmt0(r["wis"]), f"{r['rel_naive']:.2f}", f"{r['mape']:.1f}", f"{r['cov80']:.2f}", f"{r['dir_hit']:.2f}"])
table(s, 7.25, 1.8, 5.45, lbrows, [1.85, 0.75, 0.8, 0.7, 0.7, 0.65], font_size=9.5, best_row=1, row_h=0.31, num_from=1)
text(s, 7.25, 4.4, 5.45, 0.3, "COVID 前（2018–2019）最佳設定 WIS 4,262，相對 naive −36%；各分段結論一致。", size=9.5, color="n600")
finds = [[("零樣本已勝過統計基準　", {"bold": True, "color": "p900"}), (f"不加任何共變數的 TimesFM，WIS 比 naive 低 {round(100*(1-LB['tfm_expanding']['rel_naive']))}%、比 AutoETS 低 {round(100*(1-LB['tfm_expanding']['wis']/LB['ets']['wis']))}%；季節性 naive 因季節位移與 COVID 斷層失效（MASE 1.17）。", {})],
         [("假日天數是最有用的共變數　", {"bold": True, "color": "p900"}), ("加入平日假日天數與春節旗標後 1 週前 WIS 再降約 17%；季節相位、log1p、對稱平均、滑動視窗都沒有幫助。", {})],
         [("多變量聯合在每個 horizon 都最好　", {"bold": True, "color": "p900"}), (f"80% 區間涵蓋率 {best['cov80']:.2f}、60% 涵蓋率 {best['cov60']:.2f}，校準良好；高流行週中位數偏低 2–8%，屬保守。", {})]]
text(s, 0.6, 5.35, 12.1, 1.6, finds, size=11, color="n700", bullets=True, space_after=5)
footer(s, 3)
s.notes_slide.notes_text_frame.text = "圖：WIS 隨 horizon 上升，但最佳設定在每個 horizon 都低於基準。表：MAPE 10.5% 對 naive 13.1%。強調校準：涵蓋率接近名目值。"

# ---------------------------------------------------------------- slide 4 · extended validation & hit rates
s = prs.slides.add_slide(BLANK); title(s, "延伸驗證與命中率", "同一套設定在急診人次、RODS 急診%、22 縣市與 18 年齡層都優於基準；方向與閾值命中率高於 naive", kicker="03 · VALIDATION")
s.shapes.add_picture(str(SL / "backtest_h1.png"), Inches(0.6), Inches(1.7), width=Inches(7.2))
s.shapes.add_picture(str(SL / "county_rel.png"), Inches(8.3), Inches(1.7), height=Inches(4.95))
ext = [[("全國類流感急診人次（健保）　", {"bold": True, "color": "p900"}), (f"最佳同為四變量聯合＋春節＋假日：WIS {fmt0(ERB['wis'])}，相對 naive {ERB['rel_naive']:.2f}，MAPE {ERB['mape']:.1f}%，80% 涵蓋率 {ERB['cov80']:.2f}。", {})],
       [("RODS 急診類流感%　", {"bold": True, "color": "p900"}), (f"最佳 WIS {RDB['wis']:.2f} 個百分點，相對 naive {RDB['rel_naive']:.2f}；以流行閾值 10% 計，閾值命中率（敏感度）{RDB['thr_hit_rate']:.2f}、誤報率 {RDB['thr_false_alarm']:.2f}（naive 0.89 / 0.30）。", {})],
       [("縣市與年齡層聯合預測　", {"bold": True, "color": "p900"}), (f"22 縣市總 WIS 相對逐縣市 naive {CTY['mv_cov']['rel_naive']:.2f}（每個縣市都 < 1，右圖）；18 年齡層 {AGE['mv_cov']['rel_naive']:.2f}。", {})],
       [("命中率　", {"bold": True, "color": "p900"}), (f"門診最佳設定方向命中率 {best['dir_hit']:.2f}（1 週前 {POST['dir_hit_by_h'][T['best']][0]:.2f}），三分類（漲/持平/跌）{best['dir_hit3']:.2f}，±10% 容忍帶 {best['hit_tol10']:.2f}；ETS 與 Theta 的方向命中率僅 0.4 左右。", {})]]
text(s, 0.6, 5.25, 7.4, 1.7, ext, size=10.5, color="n700", bullets=True, space_after=4, line=1.1)
footer(s, 4)
s.notes_slide.notes_text_frame.text = "左圖：1 週前預測貼合實際，區間涵蓋大多數週，2025 年初春節週的極端下降只被部分捕捉。右圖：22 縣市聯合預測每個縣市都優於 naive，離島增益最小。"

# ---------------------------------------------------------------- slide 5 · current situation & latest forecast
s = prs.slides.add_slide(BLANK); title(s, f"目前疫情與最新預測（資料至 {OUT_IND['origin_yw']}，{OUT_IND['origin_date']} 起的一週）", "夏末流感波正在上升；每週預測頁頂部的判讀由程式依最新資料與分位數自動生成", kicker="04 · LATEST FORECAST")
s.shapes.add_picture(str(SL / "latest_fan.png"), Inches(0.6), Inches(1.7), width=Inches(7.3))
rect(s, 8.2, 1.7, 4.5, 1.55, fill="p50")
text(s, 8.4, 1.8, 4.1, 1.4, NAR["headline"], size=11.5, color="p900", bold=True, line=1.2)
f = OUT_IND["forecast"]
kpis = [("類流感門診 4 週中位數", " → ".join(fmt0(x["median"]) for x in f), f"第 1 週 80% 區間 {fmt0(f[0]['q10'])}–{fmt0(f[0]['q90'])}"),
        ("RODS 急診類流感%", " → ".join(f"{x['median']:.1f}" for x in next(i for i in latest["indicators"] if i["key"] == "rods_ili_pct")["forecast"]) + " %", "維持在 10% 以上的機率約 95%"),
        ("類流感急診人次", " → ".join(fmt0(x["median"]) for x in next(i for i in latest["indicators"] if i["key"] == "nhi_er_ili")["forecast"]), "高點落在第 3 週"),
        ("流感併發重症（起點 202633）", " → ".join(fmt0(x["median"]) for x in next(i for i in latest["indicators"] if i["key"] == "nidds_severe")["forecast"]) + " 例", "通報延遲約 3 週")]
for i, (lab, val, sub) in enumerate(kpis):
    y = 3.45 + i * 0.78
    text(s, 8.2, y, 4.5, 0.25, lab, size=9.5, color="n500", bold=True)
    text(s, 8.2, y + 0.24, 4.5, 0.3, val, size=13, color="n900", bold=True)
    text(s, 8.2, y + 0.53, 4.5, 0.22, sub, size=9, color="n600")
text(s, 0.6, 5.35, 7.3, 1.5, [[("目前趨勢　", {"bold": True, "color": "p900"}), (NAR["current"][0], {})], [("提醒　", {"bold": True, "color": "p900"}), (NAR["caveats"][0] + " " + NAR["caveats"][1], {})]], size=10, color="n700", space_after=4, line=1.1)
footer(s, 5)
s.notes_slide.notes_text_frame.text = "說明這是自動判讀：規則從最新面板與 9 個分位數推得走勢形狀、觸頂週、續升機率與閾值機率。提醒觀眾 1–2 週最可靠、高流行週偏保守。"

# ---------------------------------------------------------------- slide 6 · dashboard, limits, next
s = prs.slides.add_slide(BLANK); title(s, "線上 Dashboard、限制與下一步", "GitHub Pages 靜態站台，每週一個指令更新；模型權重授權與申報回補是主要注意事項", kicker="05 · DASHBOARD & NEXT")
cols = [("線上 Dashboard（GitHub Pages）", [
            [("每週預測　", {"bold": True}), ("drhao.github.io/forecast-teller/：趨勢判讀、KPI、四項指標扇形圖（60%/80% 區間）、分位數表", {})],
            [("回測　", {"bold": True}), ("/backtest.html：三個目標序列、可排序排行榜、WIS/命中率/涵蓋率依 horizon、1–4 週前預測 vs 實際、縣市與年齡層", {})],
            [("報告　", {"bold": True}), ("/report.html：一頁式解讀與評估報告（可列印）", {})],
            [("每週更新　", {"bold": True}), ("資料放入 data/ → scripts/weekly_update.sh --push（建面板、預測、重建站台、推送）", {})]]),
        ("限制與注意事項", [
            "TimesFM 3.0 權重為非商業、非生產用途授權；正式上線需改用 2.5（Apache-2.0）或另取授權。",
            "健保申報有回補，最近 1–2 週低報；管線自動排除不完整週，正式使用應以首報/終報比值校正。",
            "高流行週中位數偏低 2–8%，上升期請看區間上緣；3–4 週預測僅供規劃。",
            "回測窗只有十年，最早起點僅兩季暖機；閾值以整段資料的分位數決定，含少量事後資訊。"]),
        ("下一步", [
            "TimesFM 與 ETS 的簡單集成，看是否穩定尾端風險。",
            "健保申報回補的即時校正（nowcasting）。",
            "季節峰值時間與高度的專門評估指標。",
            "評估 TimesFM 2.5 LoRA 微調是否值得；納入更多目標（型別、年齡層即時預測）。",
            "假日檔改為每年由官方辦公日曆表自動更新。",
            "加入合約實驗室的病毒監測資料（目前實驗室資料來自 LARS）。"])]
for i, (h, items) in enumerate(cols):
    x = 0.6 + i * 4.1
    rect(s, x, 1.75, 3.9, 4.25, fill="n100" if i else "p50")
    text(s, x + 0.25, 1.9, 3.4, 0.35, h, size=14, color="p900", bold=True)
    text(s, x + 0.25, 2.4, 3.4, 3.5, items, size=11.5, color="n700", bullets=True, space_after=9, line=1.15)
text(s, 0.6, 6.25, 12.1, 0.6, [[("程式碼與流程　", {"bold": True, "color": "p900"}), ("github.com/drhao/forecast-teller（程式 Apache-2.0；原始資料不入版控）· 模型權重 google/timesfm-3.0-pytorch（非商業授權）· 回測結果與逐起點預測皆可重現（outputs/backtest/*.parquet）。", {})]], size=11, color="n700")
footer(s, 6)
s.notes_slide.notes_text_frame.text = "收尾：三個網址；每週更新流程；授權與資料回補是上線前要解決的兩件事。"

out = SL / f"forecast-teller_成果報告_{TODAY}.pptx"; prs.save(out); print("saved", out, f"{out.stat().st_size//1024} KB")
