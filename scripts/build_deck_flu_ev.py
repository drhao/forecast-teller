#!/usr/bin/env python
"""Build the 9-slide flu + enterovirus deck (cover + 4 flu + 4 EV), numbers read from docs/data, docs/ev/data and outputs/.

Run `python scripts/make_slide_charts.py` first (charts in outputs/slides/).
"""
import json, sys, datetime as dt
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deck_lib import *  # noqa: E402,F403
from forecast_teller import OUTPUT_DIR, ROOT  # noqa: E402
from forecast_teller.backtest import DEFAULT_THRESHOLDS  # noqa: E402

RTHR = DEFAULT_THRESHOLDS["rods_ili_pct"]
SL = OUTPUT_DIR / "slides"
TODAY = dt.date.today().isoformat()
FOOT = "forecast-teller · TimesFM 3.0 台灣流感與腸病毒預測 · 資料：疾管署（健保、RODS、NIDDS、LARS；資料開放平台）· 圖表依《疫情資料視覺化指引》"


def load(p): return json.loads((ROOT / p).read_text(encoding="utf-8"))


def yw_fmt(yw): return f"{yw[:4]}w{yw[4:]}"


def eval_rows(path, origin):
    e = pd.read_csv(path, dtype={"origin_yw": str, "target_yw": str})
    return e[e["origin_yw"] == origin]


def pick(e, mode, target, h):
    r = e[(e["mode"] == mode) & (e["target"] == target) & (e["h"] == h)]
    return None if r.empty else r.iloc[0]


# ---------------------------------------------------------------- flu data
bt = load("docs/data/backtest.json"); latest = load("docs/data/latest.json"); meta = load("docs/data/meta.json")
T = bt["targets"]["nhi_out_ili"]; POST = T["segments"]["post_covid"]; LB = {x["config"]: x for x in POST["leaderboard"]}
PRE = {x["config"]: x for x in T["segments"]["pre_covid"]["leaderboard"]}
ER = bt["targets"]["nhi_er_ili"]; ERB = {x["config"]: x for x in ER["segments"]["post_covid"]["leaderboard"]}[ER["best"]]
RD = bt["targets"]["rods_ili_pct"]; RDL = {x["config"]: x for x in RD["segments"]["post_covid"]["leaderboard"]}; RDB = RDL[RD["best"]]; RDN = RDL["naive"]
CTY = {x["config"]: x for x in bt["groups"]["county"]["segments"]["post_covid"]}; AGE = {x["config"]: x for x in bt["groups"]["age"]["segments"]["post_covid"]}
best, naive = LB[T["best"]], LB["naive"]
impr = [round(100 * (1 - a / b)) for a, b in zip(POST["wis_by_h"][T["best"]], POST["wis_by_h"]["naive"])]
NAR = latest["narrative"]; IND = {i["key"]: i for i in latest["indicators"]}; OUT_IND = IND["nhi_out_ili"]; COMB = latest["combined"]
COV = meta["coverage"]; BW = meta["backtest_window"]
FE = eval_rows(OUTPUT_DIR / "forecast_log/evaluation.csv", "202636")

# ---------------------------------------------------------------- EV data
ebt = load("docs/ev/data/backtest.json"); elatest = load("docs/ev/data/latest.json")
ET = ebt["targets"]["ev_oe"]; EPOST = ET["segments"]["post_covid"]; ELB = {x["config"]: x for x in EPOST["leaderboard"]}
EPRE = {x["config"]: x for x in ET["segments"]["pre_covid"]["leaderboard"]}
ebest, enaive = ELB[ET["best"]], ELB["naive"]
eimpr = [round(100 * (1 - a / b)) for a, b in zip(EPOST["wis_by_h"][ET["best"]], EPOST["wis_by_h"]["naive"])]
ENAR = elatest["narrative"]; EIND = {i["key"]: i for i in elatest["indicators"]}; EOE = EIND["ev_oe"]; EST = elatest["status"]
ECOV = json.loads((ROOT / "ev_forecast/data_processed/coverage.json").read_text(encoding="utf-8"))
THR = pd.read_csv(ROOT / "ev_forecast/data/thresholds.csv", encoding="utf-8-sig")
EE = eval_rows(ROOT / "ev_forecast/outputs/forecast_log/evaluation.csv", "202636")
yr_wis = ET["yearly_wis"]["h1"]

prs, BLANK = new_deck()

# ================================================================ 1 · cover
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, 13.333, 7.5, fill="n900")
text(s, 0.8, 1.05, 11, 0.3, "FORECAST-TELLER · 回測與預測成果 · " + TODAY, size=11, color="p300", bold=True)
text(s, 0.8, 1.5, 11.8, 1.0, "以 TimesFM 3.0 零樣本預測台灣流感與腸病毒疫情", size=34, color="n0", bold=True)
text(s, 0.8, 2.55, 11.5, 0.9, "同一套資料管線與滾動回測，兩條每週更新的線上預測：類流感門急診（流感）與門急診合計（腸病毒）", size=18, color="n300")
stats = [(f"{T['n_origins']}", f"個回測起點（{yw_fmt(T['origin_first'])}–{yw_fmt(T['origin_last'])}，每週一個）"),
         (f"−{round(100 * (1 - best['rel_naive']))}%", "流感門診最佳設定 WIS 相對 naive（COVID 後）"),
         (f"−{round(100 * (1 - ebest['rel_naive']))}%", "腸病毒門急診最佳設定 WIS 相對 naive（COVID 後）"),
         (f"{best['cov80']:.2f} / {ebest['cov80']:.2f}", "80% 預測區間實際涵蓋率（流感 / 腸病毒，理想 0.80）")]
for i, (v, lab) in enumerate(stats):
    x = 0.8 + i * 3.05
    text(s, x, 4.1, 2.9, 0.8, v, size=36, color="p300", bold=True)
    text(s, x, 4.95, 2.8, 0.75, lab, size=11, color="n300")
text(s, 0.8, 6.3, 11.8, 0.6, ["Google TimesFM 3.0（330M 參數，零樣本、未微調，一次輸出 9 個分位數）· 疾管署健保、RODS、NIDDS、LARS 週資料與資料開放平台",
                               "drhao.github.io/forecast-teller（流感）· drhao.github.io/forecast-teller/ev/（腸病毒）· 每週更新並留存預測紀錄供事後評估"], size=11, color="n400", space_after=2)
s.notes_slide.notes_text_frame.text = "兩條工作線共用核心模組與回測設計；流感用內部週資料，腸病毒用資料開放平台。重點：零樣本、可驗證、每週自動更新。"

# ================================================================ 2 · flu data & method
s = prs.slides.add_slide(BLANK); title(s, "流感：資料與方法", "五個疾管署監測來源對齊成週資料面板，TimesFM 3.0 零樣本預測，逐層加入共變數與多變量聯合", kicker="流感 01 · DATA & METHOD")
rect(s, 0.6, 1.75, 6.3, 4.05, fill="n100")
text(s, 0.85, 1.9, 5.8, 0.3, "資料來源（疫情週；頭尾不完整週自動排除）", size=12.5, color="p900", bold=True)
rows = [["來源", "內容", "完整週"],
        ["健保申報 門診/住院", "類流感（48X）、流感（487）就診人次、總人次", f"{COV['nhi']['first_complete'][:4]}–{yw_fmt(COV['nhi']['last_complete'])}"],
        ["健保申報 急診", "同上，急診", f"{COV['nhi_er']['first_complete'][:4]}–{yw_fmt(COV['nhi_er']['last_complete'])}"],
        ["RODS 即時疫情監視", "急診類流感人次、百分比（234 家醫院）", f"{COV['rods']['first_complete'][:4]}–{yw_fmt(COV['rods']['last_complete'])}"],
        ["NIDDS 法定傳染病", "流感併發重症確定病例（發病週，多排除 3 週）", f"{COV['nidds']['first_complete'][:4]}–{yw_fmt(COV['nidds']['last_complete'])}"],
        ["LARS 實驗室自動通報", "A/B 型陽性數、檢驗件數、陽性率", f"{COV['lab']['first_complete'][:4]}–{yw_fmt(COV['lab']['last_complete'])}"],
        ["RESP_LAB 社區合約實驗室", "呼吸道病原體 PCR（NIDSS 網站抓取）；只用於敘事", f"{COV['resp']['first_complete'][:4]}–{yw_fmt(COV['resp']['last_complete'])}"]]
table(s, 0.85, 2.3, 5.85, rows, [1.75, 2.95, 1.15], font_size=9.3, row_h=0.4, num_from=None)
text(s, 0.85, 5.2, 5.8, 0.55, "22 縣市 × 19 年齡層皆有分解；假日檔（春節、平日假日天數）延伸至 2026 年底；NIDDS 隱性零值自動補 0。", size=10, color="n700")
steps = [("TimesFM 3.0", "Google 時間序列基礎模型，330M 參數，零樣本、不微調；一次輸出 9 個分位數（q10–q90）。"),
         ("已知未來共變數", "春節週旗標、每週平日假日天數（最有用）；季節相位、log1p、對稱平均、滑動視窗皆無幫助。"),
         ("多變量聯合", "門診 + 急診 + RODS + 重症四個序列一起預測（變數注意力）；22 縣市 / 18 年齡層也各自聯合。"),
         ("基準模型", "last-value naive、MA3（前 3 週平均）、季節性 naive、AutoETS、Theta；分位數為中位數置中的經驗誤差。"),
         ("回測設計", f"資料窗 {yw_fmt(BW['start'])}–{yw_fmt(BW['end'])}（{BW['weeks']} 週），暖機 {BW['warmup']} 週，每週一個起點共 {BW['origins']} 個，預測 1–4 週；分 COVID 前 / 期 / 後三段。")]
for i, (h, d) in enumerate(steps):
    y = 1.8 + i * 0.82
    circle_num(s, 7.3, y + 0.02, i + 1)
    text(s, 7.8, y - 0.02, 4.9, 0.3, h, size=13, color="p900", bold=True)
    text(s, 7.8, y + 0.3, 4.9, 0.55, d, size=10.5, color="n700", line=1.1)
rect(s, 0.6, 6.0, 12.1, 0.85, fill="p50")
text(s, 0.85, 6.1, 11.7, 0.65, [[("評估指標　", {"bold": True, "color": "p900"}), (f"WIS（加權區間分數，主指標）、MAE、MAPE、MASE、80%/60% 區間涵蓋率、方向命中率、±10% 容忍帶命中率、閾值命中率與誤報率（門診以第 75 百分位、RODS 以流行閾值 {RTHR:g}%）。", {})],
                                 [("決策規則　", {"bold": True, "color": "p900"}), ("以 COVID 後（2023–2025）1–4 週平均 WIS 選定設定，並確認 COVID 前不退步。", {})]], size=10.5, color="n700", space_after=2)
footer(s, 2, FOOT)
s.notes_slide.notes_text_frame.text = "模型完全零樣本；所有比較在同一組 415 個起點上；COVID 期（2020–2022）不做決策依據但保留在 context。RESP_LAB 回測顯示當共變數反而變差，所以只用在敘事。"

# ================================================================ 3 · flu backtest
s = prs.slides.add_slide(BLANK); title(s, "流感：回測結果（全國類流感門診人次，COVID 後 2023–2025，h = 1–4）", f"最佳設定為四變量聯合 + 春節旗標 + 假日天數：WIS 較 last-value naive 低 {round(100*(1-best['rel_naive']))}%，1–4 週分別 −{impr[0]}% / −{impr[1]}% / −{impr[2]}% / −{impr[3]}%", kicker="流感 02 · BACKTEST")
s.shapes.add_picture(str(SL / "wis_by_horizon.png"), Inches(0.6), Inches(1.75), width=Inches(6.4))
lbrows = [["設定", "WIS", "相對 naive", "MAPE %", "80% 涵蓋", "方向命中"]]
for cfg, name in [(T["best"], "四變量聯合＋春節＋假日"), ("tfm_cov_cny_hol", "單變量＋春節＋假日"), ("tfm_expanding", "TimesFM zero-shot"), ("ets", "AutoETS"), ("naive", "last-value naive"), ("ma3", "MA3"), ("snaive", "季節性 naive")]:
    r = LB[cfg]; lbrows.append([name, fmt0(r["wis"]), f"{r['rel_naive']:.2f}", f"{r['mape']:.1f}", f"{r['cov80']:.2f}", f"{r['dir_hit']:.2f}"])
table(s, 7.25, 1.8, 5.45, lbrows, [1.85, 0.75, 0.8, 0.7, 0.7, 0.65], font_size=9.5, best_row=1, row_h=0.31, num_from=1)
text(s, 7.25, 4.4, 5.45, 0.3, f"COVID 前（2018–2019）最佳設定 WIS {fmt0(PRE[T['best']]['wis'])}，相對 naive −{round(100*(1-PRE[T['best']]['rel_naive']))}%；各分段結論一致。", size=9.5, color="n600")
finds = [[("零樣本已勝過統計基準　", {"bold": True, "color": "p900"}), (f"不加任何共變數的 TimesFM，WIS 比 naive 低 {round(100*(1-LB['tfm_expanding']['rel_naive']))}%、比 AutoETS 低 {round(100*(1-LB['tfm_expanding']['wis']/LB['ets']['wis']))}%；季節性 naive 因季節位移與 COVID 斷層失效（MASE {LB['snaive']['mase']:.2f}）。", {})],
         [("假日天數是最有用的共變數　", {"bold": True, "color": "p900"}), (f"加入平日假日天數與春節旗標後 WIS 再降至 naive 的 {LB['tfm_cov_cny_hol']['rel_naive']:.2f}；季節相位、log1p、對稱平均、滑動視窗都沒有幫助。", {})],
         [("多變量聯合在每個 horizon 都最好　", {"bold": True, "color": "p900"}), (f"80% 區間涵蓋率 {best['cov80']:.2f}、60% 涵蓋率 {best['cov60']:.2f}，校準良好；高流行週中位數偏低 2–8%，屬保守。", {})]]
text(s, 0.6, 5.35, 12.1, 1.6, finds, size=11, color="n700", bullets=True, space_after=5)
footer(s, 3, FOOT)
s.notes_slide.notes_text_frame.text = f"圖：WIS 隨 horizon 上升，但最佳設定在每個 horizon 都低於基準。表：MAPE {best['mape']:.1f}% 對 naive {naive['mape']:.1f}%。強調校準：涵蓋率接近名目值。"

# ================================================================ 4 · flu validation & hit rates
s = prs.slides.add_slide(BLANK); title(s, "流感：延伸驗證與命中率", "同一套設定在急診人次、RODS 急診%、22 縣市與 18 年齡層都優於基準；方向與閾值命中率高於 naive", kicker="流感 03 · VALIDATION")
s.shapes.add_picture(str(SL / "backtest_h1.png"), Inches(0.6), Inches(1.7), width=Inches(7.2))
s.shapes.add_picture(str(SL / "county_rel.png"), Inches(8.3), Inches(1.7), height=Inches(4.95))
ext = [[("全國類流感急診人次（健保）　", {"bold": True, "color": "p900"}), (f"最佳同為四變量聯合＋春節＋假日：WIS {fmt0(ERB['wis'])}，相對 naive {ERB['rel_naive']:.2f}，MAPE {ERB['mape']:.1f}%，80% 涵蓋率 {ERB['cov80']:.2f}。", {})],
       [("RODS 急診類流感%　", {"bold": True, "color": "p900"}), (f"最佳 WIS {RDB['wis']:.2f} 個百分點，相對 naive {RDB['rel_naive']:.2f}；以流行閾值 {RTHR:g}% 計，閾值命中率（敏感度）{RDB['thr_hit_rate']:.2f}、誤報率 {RDB['thr_false_alarm']:.2f}（naive {RDN['thr_hit_rate']:.2f} / {RDN['thr_false_alarm']:.2f}）。", {})],
       [("縣市與年齡層聯合預測　", {"bold": True, "color": "p900"}), (f"22 縣市總 WIS 相對逐縣市 naive {CTY['mv_cov']['rel_naive']:.2f}（每個縣市都 < 1，右圖）；18 年齡層 {AGE['mv_cov']['rel_naive']:.2f}。", {})],
       [("命中率　", {"bold": True, "color": "p900"}), (f"門診最佳設定方向命中率 {best['dir_hit']:.2f}（1 週前 {POST['dir_hit_by_h'][T['best']][0]:.2f}），三分類（漲/持平/跌）{best['dir_hit3']:.2f}，±10% 容忍帶 {best['hit_tol10']:.2f}；閾值命中率 {best['thr_hit_rate']:.2f}、誤報率 {best['thr_false_alarm']:.2f}（naive {naive['thr_hit_rate']:.2f} / {naive['thr_false_alarm']:.2f}）。", {})]]
text(s, 0.6, 5.25, 7.4, 1.7, ext, size=10.5, color="n700", bullets=True, space_after=4, line=1.1)
footer(s, 4, FOOT)
s.notes_slide.notes_text_frame.text = "左圖：1 週前預測貼合實際，區間涵蓋大多數週，2025 年初春節週的極端下降只被部分捕捉。右圖：22 縣市聯合預測每個縣市都優於 naive，離島增益最小。"

# ================================================================ 5 · flu latest forecast & real-time evaluation
s = prs.slides.add_slide(BLANK); title(s, f"流感：最新預測與即時評估（資料至 {OUT_IND['origin_yw']}，{OUT_IND['origin_date']} 起的一週）", "每週預測頁的判讀由程式依最新資料與分位數自動生成；每次預測都存檔，隔週與實際值比對", kicker="流感 04 · LATEST FORECAST")
s.shapes.add_picture(str(SL / "latest_fan.png"), Inches(0.6), Inches(1.7), width=Inches(7.3))
rect(s, 8.2, 1.7, 4.5, 1.55, fill="p50")
text(s, 8.4, 1.8, 4.1, 1.4, NAR["headline"], size=11.5, color="p900", bold=True, line=1.2)
f = OUT_IND["forecast"]
kpi_column(s, 8.2, 3.45, 4.5, [
    ("類流感門診 4 週中位數", " → ".join(fmt0(x["median"]) for x in f), f"第 1 週 80% 區間 {fmt0(f[0]['q10'])}–{fmt0(f[0]['q90'])}"),
    ("門急診合計（門診 + 急診）", " → ".join(fmt0(x["median"]) for x in COMB["forecast"]), f"起點週實際 {fmt0(COMB['last_observed'])}"),
    ("RODS 急診類流感%", " → ".join(f"{x['median']:.1f}" for x in IND["rods_ili_pct"]["forecast"]) + " %", [x for x in NAR["outlook"] if x.startswith("RODS")][0].split("；")[-1].rstrip("。")),
    (f"流感併發重症（起點 {IND['nidds_severe']['origin_yw']}）", " → ".join(fmt0(x["median"]) for x in IND["nidds_severe"]["forecast"]) + " 例", "發病週計，通報延遲約 3 週")])
o1, o2, e1, r1 = pick(FE, "joint", "nhi_out_ili", 1), pick(FE, "joint", "nhi_out_ili", 2), pick(FE, "joint", "nhi_er_ili", 1), pick(FE, "uni", "rods_ili_pct", 1)
rect(s, 0.6, 5.3, 7.3, 1.6, fill="n100")
text(s, 0.8, 5.38, 6.9, 1.5, [[("即時評估（起點 202636 的預測，2026-09-17 發布）　", {"bold": True, "color": "p900"})],
     [(f"1 週前（目標 202637）：門診預測 {fmt0(o1['median'])}、實際 {fmt0(o1['actual'])}（誤差 {100*o1['ape']:.1f}%）；急診 {fmt0(e1['median'])} 對 {fmt0(e1['actual'])}（{100*e1['ape']:.1f}%）；RODS% {r1['median']:.1f} 對 {r1['actual']:.1f}（{100*r1['ape']:.1f}%）。三者都落在 80% 區間內。", {})],
     [(f"2 週前（目標 202638）：門診預測 {fmt0(o2['median'])}、實際 {fmt0(o2['actual'])}（高估 {100*o2['ape']:.0f}%，仍在 80% 區間內）。202638 的健保總就診人次比前週少 12%，疑似申報未齊，下次更新可能上修。", {})],
     [("提醒　", {"bold": True, "color": "p900"}), (NAR["caveats"][1], {})]], size=9.5, color="n700", space_after=3, line=1.1)
footer(s, 5, FOOT)
s.notes_slide.notes_text_frame.text = "說明這是自動判讀：規則從最新面板與 9 個分位數推得走勢形狀、續升機率與閾值機率。即時評估來自 outputs/forecast_log/，每週累積。"

# ================================================================ 6 · EV data, thresholds & method
s = prs.slides.add_slide(BLANK); title(s, "腸病毒：資料、流行閾值與方法", "資料開放平台的健保門診與 RODS 急診腸病毒週人次；流行閾值依年記錄；同一套 TimesFM 3.0 零樣本回測", kicker="腸病毒 01 · DATA & METHOD")
rect(s, 0.6, 1.75, 6.3, 4.05, fill="n100")
text(s, 0.85, 1.9, 5.8, 0.3, "資料來源（資料開放平台，每週自動下載）", size=12.5, color="p900", bold=True)
rows = [["來源", "內容", "完整週"],
        ["健保申報 門診/住院", "腸病毒就診人次、總人次；9 個年齡層、22 縣市（無急診檔）", f"{ECOV['nhi']['first_complete'][:4]}–{yw_fmt(ECOV['nhi']['last_complete'])}"],
        ["RODS 急診", "腸病毒就診人次（無分母，分母借用流感面板的 RODS 總人次）", f"{ECOV['rods']['first_complete'][:4]}–{yw_fmt(ECOV['rods']['last_complete'])}"],
        ["門急診合計 ev_oe", "健保門診 + RODS 急診；急診暫以 RODS 代替健保急診", "流行期指標"]]
table(s, 0.85, 2.3, 5.85, rows, [1.55, 3.15, 1.15], font_size=9.3, row_h=0.46, num_from=None)
n_conf = int((THR["basis"] == "新聞稿確認").sum()); n_assume = int((THR["basis"] == "推定").sum())
thr26 = int(THR.loc[THR["year"] == 2026, "threshold"].iloc[0]); thr_old = int(THR.loc[THR["year"] < 2026, "threshold"].iloc[0])
text(s, 0.85, 4.2, 5.8, 1.5, [[("流行閾值（每年由疾管署設定）　", {"bold": True, "color": "p900"}), (f"{THR['year'].min()}–2025 年 {thr_old:,} 人次（{n_conf - 1} 年有新聞稿確認、{n_assume} 年推定）、2026 年 {thr26:,} 人次；記錄於 ev_forecast/data/thresholds.csv，每年手動更新。", {})],
                               [("流行期規則（暫定）　", {"bold": True, "color": "p900"}), ("單週達閾值即進入、連續 2 週低於即脫離；近年流行期：" + "、".join(f"{yw_fmt(p['start'])}–{yw_fmt(p['end'])}" for p in EST["periods"][-3:]) + "。", {})],
                               [("與新聞稿公布值比較　", {"bold": True, "color": "p900"}), ("本合成指標在 2016–2023 年偏低 2–9%，2024 年後在 −2% 至 +6% 內（新聞稿為首報值，開放資料為回補後）。", {})]], size=9.8, color="n700", space_after=3, line=1.1)
steps = [("與流感相同的骨幹", "TimesFM 3.0 零樣本、9 個分位數；同一組 415 個起點（2018–2025 每週）、同一套基準與指標。"),
         ("腸病毒專屬共變數", "學校行事曆（暑假 7/1–8/29、寒假、開學前兩週）與平日假日天數；春節旗標改為學校行事曆。"),
         ("多變量聯合", "門急診合計 + 門診 + RODS 三序列聯合（變數注意力）；加入住院無進一步增益。"),
         ("波形特徵", "一年兩波、波形逐年不同；季節性 naive 幾乎無用（WIS 約 naive 的 4 倍），所以不能靠「去年同週」。"),
         ("閾值評估", "以各年流行閾值計閾值命中率（敏感度）與誤報率，直接對應流行期判定的實務需求。")]
for i, (h, d) in enumerate(steps):
    y = 1.8 + i * 0.82
    circle_num(s, 7.3, y + 0.02, i + 1)
    text(s, 7.8, y - 0.02, 4.9, 0.3, h, size=13, color="p900", bold=True)
    text(s, 7.8, y + 0.3, 4.9, 0.55, d, size=10.5, color="n700", line=1.1)
rect(s, 0.6, 6.0, 12.1, 0.85, fill="p50")
text(s, 0.85, 6.1, 11.7, 0.65, [[("待確認　", {"bold": True, "color": "alert"}), ("閾值與流行期規則來自新聞稿，尚待與疾管署定義核對；取得健保急診腸病毒檔後，急診部分應改用健保急診並重跑回測。", {})],
                                 [("資料開放平台　", {"bold": True, "color": "p900"}), ("od.cdc.gov.tw 憑證鏈缺中繼憑證，下載腳本已內建修法；每週 refresh_data.sh 自動下載。", {})]], size=10.5, color="n700", space_after=2)
footer(s, 6, FOOT)
s.notes_slide.notes_text_frame.text = "強調兩點差異：急診用 RODS 代替，閾值依年手動記錄。流行期規則與閾值都標「待確認」。"

# ================================================================ 7 · EV backtest
s = prs.slides.add_slide(BLANK); title(s, "腸病毒：回測結果（門急診合計，COVID 後 2023–2025，h = 1–4）", f"最佳設定為三變量聯合 + 學校行事曆 + 假日天數：WIS 較 last-value naive 低 {round(100*(1-ebest['rel_naive']))}%，1–4 週分別 −{eimpr[0]}% / −{eimpr[1]}% / −{eimpr[2]}% / −{eimpr[3]}%", kicker="腸病毒 02 · BACKTEST")
s.shapes.add_picture(str(SL / "ev_wis_by_horizon.png"), Inches(0.6), Inches(1.75), width=Inches(6.4))
lbrows = [["設定", "WIS", "相對 naive", "MAPE %", "80% 涵蓋", "方向命中"]]
for cfg, name in [(ET["best"], "三變量聯合＋學校＋假日"), ("tfm_cov_school_hol", "單變量＋學校＋假日"), ("tfm_expanding", "TimesFM zero-shot"), ("ets", "AutoETS"), ("naive", "last-value naive"), ("ma3", "MA3"), ("snaive", "季節性 naive")]:
    r = ELB[cfg]; lbrows.append([name, fmt0(r["wis"]), f"{r['rel_naive']:.2f}", f"{r['mape']:.1f}", f"{r['cov80']:.2f}", f"{r['dir_hit']:.2f}"])
table(s, 7.25, 1.8, 5.45, lbrows, [1.85, 0.75, 0.8, 0.7, 0.7, 0.65], font_size=9.5, best_row=1, row_h=0.31, num_from=1)
text(s, 7.25, 4.4, 5.45, 0.3, f"COVID 前（2018–2019）最佳設定 WIS {fmt0(EPRE[ET['best']]['wis'])}，相對 naive −{round(100*(1-EPRE[ET['best']]['rel_naive']))}%；結論一致。", size=9.5, color="n600")
finds = [[("零樣本單變量已比基準好　", {"bold": True, "color": "p900"}), (f"無共變數的 TimesFM WIS 為 naive 的 {ELB['tfm_expanding']['rel_naive']:.2f}、AutoETS {ELB['ets']['rel_naive']:.2f}、MA3 {ELB['ma3']['rel_naive']:.2f}；季節性 naive {ELB['snaive']['rel_naive']:.2f}（去年同週完全不可用）。", {})],
         [("共變數與聯合的增益比流感小　", {"bold": True, "color": "p900"}), (f"學校行事曆 + 假日把 WIS 降到 naive 的 {ELB['tfm_cov_school_hol']['rel_naive']:.2f}，三變量聯合再到 {ebest['rel_naive']:.2f}；MAPE {ebest['mape']:.1f}%（naive {enaive['mape']:.1f}%）。", {})],
         [("區間偏窄　", {"bold": True, "color": "p900"}), (f"80% 涵蓋率 {ebest['cov80']:.2f}、60% 涵蓋率 {ebest['cov60']:.2f}，低於名目值；解讀時區間要放寬一些，區間校正列為下一步。", {})]]
text(s, 0.6, 5.35, 12.1, 1.6, finds, size=11, color="n700", bullets=True, space_after=5)
footer(s, 7, FOOT)
s.notes_slide.notes_text_frame.text = "腸病毒波形逐年不同，季節性 naive 失效是這條線最重要的背景。零樣本已贏基準，共變數再補一點。涵蓋率偏窄要講清楚。"

# ================================================================ 8 · EV threshold hits & h1 vs actual
s = prs.slides.add_slide(BLANK); title(s, "腸病毒：流行閾值命中與 1 週前預測 vs 實際", "以各年流行閾值評估「下週是否達閾值」的命中率與誤報率；逐年 WIS 顯示 2024 年高峰季最難", kicker="腸病毒 03 · VALIDATION")
s.shapes.add_picture(str(SL / "ev_backtest_h1.png"), Inches(0.6), Inches(1.7), width=Inches(7.2))
hit = [["設定", "閾值命中率", "誤報率", "方向命中", "三分類", "±10%"]]
for cfg, name in [(ET["best"], "三變量聯合＋學校＋假日"), ("tfm_cov_school_hol", "單變量＋學校＋假日"), ("tfm_expanding", "TimesFM zero-shot"), ("ets", "AutoETS"), ("naive", "last-value naive")]:
    r = ELB[cfg]; hit.append([name, f"{r['thr_hit_rate']:.2f}", f"{r['thr_false_alarm']:.2f}", f"{r['dir_hit']:.2f}", f"{r['dir_hit3']:.2f}", f"{r['hit_tol10']:.2f}"])
table(s, 8.1, 1.75, 4.6, hit, [1.6, 0.75, 0.6, 0.6, 0.55, 0.5], font_size=9, best_row=1, row_h=0.3, num_from=1)
text(s, 8.1, 3.65, 4.6, 0.3, f"閾值 = 各年流行閾值（回測窗內均為 {ET['threshold']:,.0f} 人次），COVID 後 h = 1–4 平均。", size=9, color="n600")
years = ["2023", "2024", "2025"]
yrows = [["1 週前 WIS", *years], ["最佳設定", *[fmt0(yr_wis[ET["best"]][y]) for y in years]], ["last-value naive", *[fmt0(yr_wis["naive"][y]) for y in years]], ["AutoETS", *[fmt0(yr_wis["ets"][y]) for y in years]]]
table(s, 8.1, 4.1, 4.6, yrows, [1.6, 1.0, 1.0, 1.0], font_size=9.5, best_row=1, row_h=0.3, num_from=1)
ext = [[("閾值命中　", {"bold": True, "color": "p900"}), (f"最佳設定閾值命中率 {ebest['thr_hit_rate']:.2f}、誤報率 {ebest['thr_false_alarm']:.2f}；naive 為 {enaive['thr_hit_rate']:.2f} / {enaive['thr_false_alarm']:.2f}。COVID 前更高（命中 {EPRE[ET['best']]['thr_hit_rate']:.2f}、誤報 {EPRE[ET['best']]['thr_false_alarm']:.2f}）。", {})],
       [("方向　", {"bold": True, "color": "p900"}), (f"方向命中率 {ebest['dir_hit']:.2f}，統計基準只有 0.33–0.40（接近隨機）；三分類 {ebest['dir_hit3']:.2f}、±10% 容忍帶 {ebest['hit_tol10']:.2f}。", {})],
       [("左圖　", {"bold": True, "color": "p900"}), ("1 週前中位數貼合實際，紅色虛線為流行閾值；上升段起點（2024 年春、2025 年秋）略為落後 1 週，峰值略偏低。", {})]]
text(s, 0.6, 5.3, 7.4, 1.6, ext, size=10.5, color="n700", bullets=True, space_after=4, line=1.1)
footer(s, 8, FOOT)
s.notes_slide.notes_text_frame.text = "閾值命中率就是「下週會不會進入流行期」的敏感度；誤報率是低於閾值卻預測達標的比例。2024 年是回測窗內最高的腸病毒年，WIS 最高。"

# ================================================================ 9 · EV latest forecast & real-time evaluation
s = prs.slides.add_slide(BLANK); title(s, f"腸病毒：最新預測與即時評估（資料至 {EOE['origin_yw']}，{EOE['origin_date']} 起的一週）", "門急診合計對流行閾值的機率判讀由程式自動生成；起點 202636 的預測已與兩週實際值比對", kicker="腸病毒 04 · LATEST FORECAST")
s.shapes.add_picture(str(SL / "ev_latest_fan.png"), Inches(0.6), Inches(1.7), width=Inches(7.3))
rect(s, 8.2, 1.7, 4.5, 1.55, fill="p50")
text(s, 8.4, 1.8, 4.1, 1.4, ENAR["headline"], size=11.5, color="p900", bold=True, line=1.2)
ef = EOE["forecast"]
import re
probs = re.findall(r"約 (\d+)%", [x for x in ENAR["outlook"] if x.startswith("達到流行閾值")][0])
ages = re.findall(r"(\d+~\d+\+?) 歲 (\d+)%（4 週變化 ([+\-−]\d+%)）", ENAR["current"][2])
kpi_column(s, 8.2, 3.45, 4.5, [
    ("門急診合計 4 週中位數", " → ".join(fmt0(x["median"]) for x in ef), f"第 1 週 80% 區間 {fmt0(ef[0]['q10'])}–{fmt0(ef[0]['q90'])}；起點週實際 {fmt0(EOE['last_observed'])}"),
    (f"達流行閾值 {thr26:,} 人次的機率（第 1–4 週）", " → ".join(f"{p}%" for p in probs), "依 9 個分位數內插；4 週內中位數都未達閾值"),
    ("健保門診 4 週中位數", " → ".join(fmt0(x["median"]) for x in EIND["ev_out"]["forecast"]), "RODS 急診 " + " → ".join(fmt0(x["median"]) for x in EIND["ev_rods"]["forecast"]) + " 人次（聯合預測的另外兩個序列）"),
    ("門診年齡分布（起點週）", "、".join(f"{a.replace('~', '–')} 歲 {p}%" for a, p, _ in ages), "4 週變化 " + " / ".join(c for _, _, c in ages) + "；開學後學齡兒童上升最快")])
j1, j2 = pick(EE, "joint", "ev_oe", 1), pick(EE, "joint", "ev_oe", 2)
rect(s, 0.6, 5.3, 7.3, 1.6, fill="n100")
text(s, 0.8, 5.38, 6.9, 1.5, [[("即時評估（起點 202636 的預測，2026-09-19 發布）　", {"bold": True, "color": "p900"})],
     [(f"當時預測門急診合計會由 11,113 升到 12,866、第 4 週達閾值機率約 60%。1 週前（目標 202637）：預測 {fmt0(j1['median'])}、實際 {fmt0(j1['actual'])}（誤差 {100*j1['ape']:.1f}%，在 80% 區間內）；2 週前（目標 202638）：預測 {fmt0(j2['median'])}、實際 {fmt0(j2['actual'])}（高估 {100*j2['ape']:.0f}%，落在 80% 區間外）。", {})],
     [("解讀　", {"bold": True, "color": "p900"}), ("上升趨勢在 202638 中斷，模型對轉折的反應慢 1 週，與回測結論一致；202638 的健保總就診人次比前週少 12%，門診腸病毒人次可能有一部分是申報未齊。", {})],
     [("提醒　", {"bold": True, "color": "p900"}), (ENAR["caveats"][1].split("：")[0] + "；" + ENAR["caveats"][3], {})]], size=9.5, color="n700", space_after=3, line=1.1)
footer(s, 9, FOOT)
s.notes_slide.notes_text_frame.text = "這是腸病毒第一筆真正的即時評估：1 週前準、2 週前高估，因為上升被打斷。預測紀錄每週累積在 ev_forecast/outputs/forecast_log/。"

out = SL / f"forecast-teller_流感與腸病毒成果_{TODAY}.pptx"; prs.save(out); print("saved", out, f"{out.stat().st_size//1024} KB")
