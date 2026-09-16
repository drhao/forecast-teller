# 以 Google TimesFM 3.0 進行台灣流感疫情預測：作法規劃

日期：2026-09-17　專案：`forecast-teller`　硬體：Apple M1 / 16 GB / Python 3.11.6

---

## 0. 一頁摘要

- **策略**：先做 zero-shot 單變量基準，再逐層加入「已知未來共變數」（春節、季節相位）、「僅過去共變數」（實驗室 A/B 型陽性、急診 RODS）、以及多變量聯合預測（縣市、年齡層、門診/急診/重症同時預測）。微調（fine-tuning）列為選配，最後才做。
- **預測目標**：全國每週類流感門診就診人次與就診率（主）、RODS 急診類流感就診百分比（主，即時性最高）、流感併發重症週病例數與縣市/年齡層分解（次）。
- **預測長度**：1–4 週為核心（每週滾動更新），8–12 週供季節規劃；同時輸出中位數與 9 個分位數（0.1–0.9），做 80% 預測區間。
- **評估**：回測資料窗固定為 **2016–2025 十年**（2016w01–2025w47，共 516 週，所有資料來源在此區間都完整）。以 2016–2017 為暖機 context，2018w01–2025w43 每週為起點做擴張視窗回測（408 個起點），以 WIS、MAE、MASE、區間涵蓋率評分，對照季節性 naive、ETS 等基準；結果分 COVID 前（2018–2019）、COVID 期（2020–2022）、COVID 後（2023–2025）三段報告。
- **環境**：`pip install "timesfm[mlx]"`（Apple Silicon 原生，與 PyTorch 版數值一致）；備案 `timesfm[torch]` 走 CPU。
- **關鍵限制**：TimesFM 3.0 權重授權為**非商業、非生產用途**；若要正式上線需改用 2.5（Apache-2.0）或另談授權。

---

## 1. TimesFM 3.0 已核對的事實（來自 README、PyPI、原始碼）

| 項目 | 內容 |
|---|---|
| 套件 | `timesfm` 3.0.2（2026-09-09 發布），Python ≥ 3.10；extras：`torch`、`mlx`、`flax`、`xreg` |
| 權重 | `google/timesfm-3.0-pytorch`（Hugging Face，未設 gated，約 1.32 GB safetensors），330M 參數、20 層、d=1280 |
| 授權 | 程式碼 Apache-2.0；**3.0 權重 `timesfm-non-commercial-license-v1.0`**（非商業、非生產）；2.5 權重仍為 Apache-2.0 |
| 匯入 | PyTorch：`from timesfm3 import TimesFM3Forecaster, TimesFM3Evaluator, ModelConfig`；MLX：`from timesfm3.mlx import TimesFM3Forecaster` |
| 建立 | `TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device=None, per_core_batch_size=4)`；device 可為 `"cuda"`/`"cpu"`（`"mps"` 會傳入 `torch.device`，官方未宣稱支援） |
| 推論 | `predict(context, horizon, past_only_covariates=None, past_future_covariates=None, return_quantiles=False, use_symmetric_averaging=False, make_positive=False, sort_quantiles=True, use_znorm=False, padding_mode="none")`；`predict_batch(contexts=[...], ...)` 接受不等長序列 |
| 輸入形狀 | 單變量 `(L,)`；多變量 `(V, L)`；僅過去共變數 `(K, L)`；過去+未來共變數 `(W, L+H)` |
| 輸出形狀 | `forecast` `(H,)` 或 `(V, H)`；`quantiles` `(H, 9)` 或 `(V, H, 9)`，分位數 0.1…0.9，**中位數在 index 4** |
| Patch | 輸入 patch 32、輸出 patch 64；horizon 任意長度，內部補到 64 的倍數後裁切；一次 forward 即產出整段（非自迴歸） |
| Context | 最長 15,360 點；短於 32 的倍數自動左側補零並遮罩 |
| NaN | 前導 NaN 會被截掉；中間 NaN 線性插補；**尾端 NaN 不處理，須自行移除** |
| 正規化 | 模型內建 RevIN，**不需外部標準化**；`use_znorm` 為選配 |
| 變數上限 | 一次 forward 最多 32 個變數（目標 + 共變數通道）；`TimesFM3Evaluator` 會自動分塊並預設 `use_symmetric_averaging=True, make_positive=True` |
| 微調 | 官方 LoRA 範例（HF Transformers + PEFT）**只針對 2.5**；3.0 目前無官方微調腳本 |
| 範例 | `timesfm3-usage/notebooks/promotion_and_synthetic_parity_demo.ipynb`：以 0/1 促銷旗標當 past-future covariate，與本案「春節週旗標」用法相同 |

---

## 2. 現有資料盤點與品質發現

### 2.1 檔案清單（`data/`）

| 檔案 | 內容 | 粒度 | 範圍（週） | 備註 |
|---|---|---|---|---|
| `NHI_PROCESSED_48X.csv` | 健保**類流感**就診人次、總就診人次 | 週 × 門診/住院 × 19 年齡層 × 22 縣市 | 200501–202548 | 主目標來源；83.7 萬列 |
| `NHI_PROCESSED_48X_ER.csv` | 健保類流感**急診** | 週 × 年齡層 × 縣市 | 201552–202549 | |
| `NHI_PROCESSED_487.csv` / `_ER.csv` | 健保**流感及其所致肺炎**（ICD 487） | 同上 | 2005– / 2016– | 較嚴格的流感定義 |
| `RODS_RS.csv` | 即時疫情監視（RODS）急診類流感、急診總人次 | 週 × 年齡層 × 234 家醫院 | 200901–202548 | 256 萬列；疾管署流行閾值指標來源 |
| `NIDDS_487A.csv` | 流感併發重症確定病例 | 發病週 × 縣市 × 鄉鎮 × 性別 × 年齡層 × 境外 | 200302–202548 | 週中位數 17 例，低計數 |
| `INFLUENZA_TYPE_YW.csv` | 合約實驗室 A 型 / B 型 / 未分型陽性數 | 週 × 縣市 × 鄉鎮 × 68 家醫院 | 201453–202603 | 型別領先指標 |
| `INFLUENZA_MON_TYPE_YW.csv` | 檢驗件數（分母） | 週 × 醫院 × 健保代碼 | 201453–202603 | 可算陽性率 |
| `date_week_mapping.csv` | 日期 → 年/週 | 日 | 1997–2100 | 見 2.2 |
| `tw_holiday.csv` | 台灣假日 | 日 | 2008–2025 | UTF-8；需延伸到 2026–2027 |

### 2.2 品質發現（會影響建模）

1. **編碼**：除 `tw_holiday.csv`、`date_week_mapping.csv` 外全部是 Big5，讀取用 `encoding="cp950"`。
2. **週定義**：2009 年起為疾管署疫情週（週日起算，2009/2014/2020/2025 有第 53 週）。**2008 年以前是以 1 月 1 日為第 1 週起點**：200501 只有 1 天、200653 只有 1 天、200753 只有 2 天。回測資料窗已定為 2016–2025（§5），因此不受影響；日後若要把 context 往前延長，最多延到 200901。
3. **頭尾不完整週**：各檔最後一週（202548、202549、202603）只有正常值的 1/5 左右，是尚未申報完的週，**必須丟棄**，否則模型會看到人為斷崖。
4. **縣市名稱**：NIDDS 含 `台中市(舊)`、`台南市(舊)`、`高雄縣(舊)`（2010 縣市合併前），需併入現制；實驗室資料有 `其他`。
5. **COVID 結構斷裂**：類流感門診人次 2019 年 366 萬 → 2021 年 106 萬；重症 2021 年僅 1 例。2023 年起反彈，**2024（437 萬）、2025（468 萬）為 21 年來最高**。季節峰值多在第 1–8 週，但 2017 在第 26 週、2022 在第 44 週、2023 在第 39 週，峰值時間並不穩定。
6. **春節效應極強**：202505（春節週）門診總人次 192 萬，前後週為 737 萬與 694 萬；類流感人次同步腰斬再反彈。這是最值得放入的「已知未來共變數」。
7. **稀疏度**：22 縣市週序列大多無零值（連江縣例外，148 個零週）；縣市 × 年齡層 418 條序列中約 18% 中位數 < 5，不適合逐條預測，改為分層或聚合。
8. **資料時效**：資料止於 2025 年 11 月至 2026 年 1 月，今天是 2026-09-17，需先更新（見 §6.3）。
9. **重症資料的隱性零值**：NIDDS 是病例列表彙總，沒有病例的週不會出現任何列；2016–2025 窗內有 136 週如此（集中在 2020–2022）。建面板時要補 0，不能當缺值交給模型插補。

---

## 3. 預測問題定義

| 目標 | 序列 | 為何 |
|---|---|---|
| A（主） | 全國每週類流感**門診**就診人次；同時預測總就診人次以推得就診率 | 疾管署流感速訊的主指標；歷史最長 |
| B（主） | RODS **急診類流感就診百分比**（全國） | 疾管署流行期判定指標；申報延遲最小，適合每週即時預測 |
| C（次） | 流感併發重症週病例數 | 低計數、落後指標；使用 `make_positive=True`，可考慮 log1p |
| D（次） | 22 縣市類流感門診人次（多變量聯合） | 區域預警；一次 forward ≤ 32 變數剛好放得下 |
| E（次） | 19 年齡層（多變量聯合） | 校園群聚、高齡風險 |
| F（輔助） | A 型 / B 型陽性數與陽性率 | 型別轉換的領先訊號，主要當共變數 |

- **Horizon**：1–4 週為核心評估與每週發布；8–12 週用於季節規劃並標示不確定性擴大。
- **輸出**：中位數、q10–q90 各分位數、80% 與 60% 預測區間、由分位數路徑推得的「未來 4 週是否超過流行閾值」機率、季節峰值週與峰值高度的估計。

---

## 4. 建模策略（四層，逐層驗證再往上加）

### Layer 1：Zero-shot 單變量基準
- 直接餵原始計數（模型內建 RevIN），`make_positive=True`，`return_quantiles=True`。
- 要比較的設定：
  - context 視窗：自 2016 起的擴張視窗 vs 近 5 季滑動視窗（260 週）vs 近 3 季滑動視窗（156 週，等同只用 COVID 後）。
  - `use_symmetric_averaging` 開/關（Evaluator 預設開，會使推論量翻倍）。
  - 原始值 vs `log1p` 外部轉換（計數資料常見；轉回時用分位數的 `expm1`）。
  - 2020–2022 保留 vs 設為 NaN（模型會線性插補，等同抹平 COVID 年）。

### Layer 2：已知未來共變數（`past_future_covariates`，形狀 `(W, L+H)`）
- **春節週旗標**（0/1）及「該週非週末假日天數」：由 `tw_holiday.csv` 產生，需補 2026–2027 年行政機關辦公日曆。
- **季節相位**：`sin/cos(2π·週序/52.18)` 兩個通道，讓模型跨越 COVID 斷層仍能對齊年週期。
- 選配：寒暑假旗標、年齡層專用的開學週旗標。

### Layer 3：僅過去共變數與多變量
- **僅過去共變數**（`past_only_covariates`，形狀 `(K, L)`）：實驗室 A/B 型陽性數與陽性率、RODS 急診百分比（預測 NHI 目標時）、重症數。**必須依實際申報延遲往後平移**（例如實驗室資料延遲 1 週就平移 1 週），否則回測會偷看未來。
- **多變量聯合預測**（`(V, L)`）：
  - 組合一：`[類流感門診, 類流感急診, RODS 急診%, 重症]`，讓變數注意力學習門診 → 急診 → 重症的傳導。
  - 組合二：22 縣市聯合（加上 2–3 個共變數通道仍 ≤ 32）。
  - 組合三：19 年齡層聯合。
  - 超過 32 變數時改用 `TimesFM3Evaluator`（自動分塊）或 `univariate=True`。

### Layer 4（選配）：微調、集成、即時修正
- **微調**：官方 LoRA 範例只支援 2.5（`google/timesfm-2.5-200m-transformers`，Apache-2.0，可商用）。3.0 需自寫訓練迴圈（分位數損失），投入前先確認 Layer 1–3 的誤差是否真的卡在領域差異。
- **集成**：TimesFM 中位數與季節性 naive / ETS 做簡單平均或依近期 WIS 加權，通常能穩定尾端風險。
- **即時修正（nowcasting）**：健保申報有回補，最近 1–2 週的數字會往上修；可用歷史「首報 / 終報」比值先校正再餵模型。

---

## 5. 評估設計（回測）

### 5.1 資料窗：2016–2025 十年（已定案）

| 項目 | 設定 |
|---|---|
| 資料窗 | 2016w01（2016-01-03）至 2025w47（2025-11-22），共 **516 週**；2025w48 起為不完整週，不用 |
| 為何從 2016 起 | 健保急診（201552 起）、合約實驗室（201453 起）、RODS、NIDDS、健保門診/住院在此區間**全部完整、無缺週**，可建立單一對齊的多來源面板 |
| 每年週數 | 2016–2019 各 52、2020 為 53、2021–2024 各 52、2025 為 47 |
| 2016 以前資料 | 不進入回測；程式保留 `context_start` 參數，日後要延長 context 只需改一個值 |

### 5.2 滾動起點（擴張視窗）

- **暖機 context**：2016w01–2017w52（104 週、兩個流感季），讓模型至少看過兩次冬季峰值。
- **起點**：2018w01 至 2025w43，每週一個起點，共 **408 個起點**（2018–2024 每年 52 或 53 個，2025 年 43 個）。每個起點只用截至該週的資料，預測 h = 1…4（另跑 h = 8）。
- **視窗變體**：擴張視窗（預設）、近 5 季滑動視窗（260 週）、近 3 季滑動視窗（156 週）。
- **計算量**：一個設定 = 408 次 predict（`predict_batch` 一次送入 408 條不等長 context），MLX 後端在 M1 上為秒到分鐘等級；開對稱平均會加倍。

### 5.3 報告分段

| 分段 | 起點年份 | 起點數 | 用途 |
|---|---|---|---|
| COVID 前 | 2018–2019 | 104 | 正常季節下的表現 |
| COVID 期 | 2020–2022 | 157 | 極低流行期，檢查是否過度預測、區間是否過寬 |
| COVID 後 | 2023–2025 | 147 | 反彈與歷史新高季，**主要決策依據** |

三段各自列出指標，再給全期平均；不把 COVID 期的分數混入決策，但也不刪除它。

### 5.4 指標

- **點預測**：MAE、WAPE、MASE（相對季節性 naive）。
- **機率預測**：**WIS**（FluSight 慣用，由 9 個分位數組成 4 個對稱區間 + 中位數）、80% 區間涵蓋率（目標 ≈ 80%）、60% 涵蓋率。
- **流行病學指標**（以季為單位，2016/17–2024/25 共 9 個完整季）：峰值週誤差、峰值高度誤差、跨越流行閾值的時點誤差。

### 5.5 基準模型

季節性 naive（去年同週，2017 起可用）、last value、ETS / Theta（`statsforecast`，每個起點重新配適於同一資料窗）；有餘裕再加 TimesFM 2.5、Chronos-2。

### 5.6 防止洩漏

尾端不完整週先刪；僅過去共變數依申報延遲平移；NIDDS 無列的週補 0；如有歷史快照（as-of 資料）優先使用。

### 5.7 決策規則

以 **COVID 後三年（2023–2025 起點）h = 1–4 的平均 WIS** 選定預設設定，並確認在 COVID 前（2018–2019）沒有明顯退步、在 COVID 期沒有離譜的過度預測。

---

## 6. 系統與程式架構

### 6.1 建議目錄

```
forecast-teller/
├── data/                      # 原始 CSV（Big5）
├── data_processed/            # 整理後的 parquet（週 × 指標 × 地區 × 年齡層）
├── src/forecast_teller/
│   ├── io.py                  # cp950 讀取、週→日期對應、丟棄不完整週、縣市名稱正規化
│   ├── panel.py               # 建立整齊週資料面板
│   ├── covariates.py          # 春節/假日、季節相位、實驗室與 RODS 共變數（含延遲平移）
│   ├── model_timesfm3.py      # 模型載入（mlx/torch）、predict 包裝、log1p 選項
│   ├── baselines.py           # 季節性 naive、ETS/Theta
│   ├── backtest.py            # 滾動起點、WIS/MAE/MASE、涵蓋率
│   └── report.py              # 扇形圖、縣市表、閾值機率
├── scripts/
│   ├── refresh_data.sh        # 從 od.cdc.gov.tw 更新
│   └── run_weekly.py          # 每週產出 CSV + 圖
├── notebooks/
├── outputs/
├── PLAN_timesfm3_flu.md
└── pyproject.toml
```

### 6.2 環境建置（M1，主線用 MLX）

```bash
cd /Users/drhao/Documents/GitHub/forecast-teller
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install "timesfm[mlx]" pandas pyarrow matplotlib statsforecast
```

備案（PyTorch 版，走 CPU）：

```bash
pip install "timesfm[torch]"
```

煙霧測試（會自 Hugging Face 下載約 1.3 GB 權重到 `~/.cache/huggingface/`）：

```bash
python -c "import numpy as np; from timesfm3.mlx import TimesFM3Forecaster as F; f=F.from_pretrained('google/timesfm-3.0-pytorch'); o=f.predict(np.sin(np.linspace(0,40,512)).astype('float32'), horizon=4, return_quantiles=True); print(o.forecast.shape, o.quantiles.shape)"
```

### 6.3 資料更新來源（疾管署資料開放平台 CKAN）

```bash
mkdir -p data/raw
curl -sSLo data/raw/NHI_Influenza_like_illness.csv https://od.cdc.gov.tw/eic/NHI_Influenza_like_illness.csv
curl -sSLo data/raw/NHI_Influenza.csv            https://od.cdc.gov.tw/eic/NHI_Influenza.csv
curl -sSLo data/raw/RODS_Influenza_like_illness.csv https://od.cdc.gov.tw/eic/RODS_Influenza_like_illness.csv
```

流感併發重症與實驗室型別資料：https://data.cdc.gov.tw/group/flu 與 https://nidss.cdc.gov.tw 。下載檔編碼可能是 UTF-8（與現有 cp950 檔不同），讀取時先偵測。

---

## 7. 程式骨架（API 已對照 3.0.2 原始碼）

```python
import numpy as np, pandas as pd
from timesfm3.mlx import TimesFM3Forecaster   # PyTorch 版：from timesfm3 import TimesFM3Forecaster

# 1) 全國每週類流感門診人次（Big5）
df = pd.read_csv("data/NHI_PROCESSED_48X.csv", encoding="cp950", dtype=str)
df = df[df["就診類別"] == "門診"]
df["ili"] = pd.to_numeric(df["類流感健保就診人次"])
df["tot"] = pd.to_numeric(df["健保就診總人次"])
wk = df.groupby("週")[["ili", "tot"]].sum().sort_index()
wk = wk.loc["201601":"202547"]          # 回測資料窗 2016–2025，共 516 週；202548 為不完整週
y = wk["ili"].to_numpy(np.float32)

# 2) 已知未來共變數：季節相位 + 春節週旗標，長度 L+H
H = 4
weeks = list(wk.index) + next_weeks(wk.index[-1], H)      # 依 date_week_mapping.csv 往後推
woy = np.array([int(w[4:]) for w in weeks], dtype=np.float32)
season = np.stack([np.sin(2*np.pi*woy/52.18), np.cos(2*np.pi*woy/52.18)])
cny = cny_week_flag(weeks)[None, :]                       # 由 tw_holiday.csv 產生，需補 2026–2027
pf = np.concatenate([season, cny]).astype(np.float32)     # (3, L+H)

# 3) 預測：中位數 + 9 分位數
fc = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")
out = fc.predict(y, horizon=H, past_future_covariates=pf,
                 return_quantiles=True, use_symmetric_averaging=True, make_positive=True)
median, q = out.forecast, out.quantiles                   # (H,), (H, 9)；q[:,0]=q10, q[:,4]=q50, q[:,8]=q90

# 4) WIS（FluSight 定義；4 個對稱區間 + 中位數）
def wis(y_true, q):
    total = 0.5 * np.abs(y_true - q[:, 4])
    for lo, hi, a in [(0, 8, 0.2), (1, 7, 0.4), (2, 6, 0.6), (3, 5, 0.8)]:
        l, u = q[:, lo], q[:, hi]
        IS = (u - l) + (2/a)*(l - y_true)*(y_true < l) + (2/a)*(y_true - u)*(y_true > u)
        total += (a/2) * IS
    return total / 4.5

# 5) 滾動起點回測：暖機 104 週（2016–2017），起點 2018w01…2025w43 共 408 個，一次 predict_batch
origins = range(104, len(y) - H)
outs = list(fc.predict_batch(
    contexts=[y[:t] for t in origins], horizon=H,
    past_future_covariates=[pf[:, :t+H] for t in origins],
    return_quantiles=True, use_symmetric_averaging=True, make_positive=True))
scores = np.array([wis(y[t:t+H], o.quantiles) for t, o in zip(origins, outs)])   # (408, H)
years = np.array([int(wk.index[t][:4]) for t in origins])
for name, (y0, y1) in {"COVID前": (2018, 2019), "COVID期": (2020, 2022), "COVID後": (2023, 2025)}.items():
    sel = (years >= y0) & (years <= y1)
    print(name, "WIS by horizon:", scores[sel].mean(axis=0))
```

多變量（22 縣市聯合）只需把 `y` 換成 `(22, L)` 陣列，輸出變為 `(22, H)` 與 `(22, H, 9)`；超過 32 變數時改用 `TimesFM3Evaluator`。

---

## 8. 里程碑

| 階段 | 工作 | 預估 | 完成判準 |
|---|---|---|---|
| 0 | 建環境、下載權重、煙霧測試、畫出全國類流感 4 週預測 | 0.5 天 | M1 上可跑，圖面正常 |
| 1 | 資料管線：cp950 讀取、週對應、丟棄不完整週、縣市正規化、共變數表到 2027、資料更新腳本 | 1–2 天 | `data_processed/` 產出 + QA 報告 |
| 2 | Layer 1 回測（2016–2025 資料窗、408 個起點）：視窗變體、log1p、對稱平均、COVID 年處理；與基準比較 | 2–3 天 | 三段分層排行榜（WIS / MAE / 涵蓋率，h = 1–4），選定預設設定 |
| 3 | Layer 2–3：春節 + 季節相位、實驗室與 RODS 共變數、多變量組合 | 2–3 天 | 量化各共變數的增益，決定保留哪些 |
| 4 | 產出：每週腳本、扇形圖、縣市表、閾值機率、README | 1–2 天 | 一鍵從更新資料到報表 |
| 5（選配） | 微調（2.5 LoRA 或 3.0 自寫）、集成、即時修正 | 1–2 週 | 相對階段 3 的 WIS 改善 |

---

## 9. 風險與對策

| 風險 | 影響 | 對策 |
|---|---|---|
| 3.0 權重非商業、非生產授權 | 無法用於正式公衛系統 | 研究用先行；上線前改 2.5（Apache-2.0）或取得授權；程式碼層面把模型版本做成可切換 |
| COVID 斷層與 2024–2025 歷史新高 | 模型對峰值高度可能低估或高估 | 回測分 COVID 前/期/後三段報告；比較視窗變體；加入季節相位共變數 |
| 資料窗只有十年，最早的起點僅兩季暖機 | 2018 年起點的 context 較短，分數可能偏差 | 報告時另列 2018 單年；必要時把暖機延長到三季（起點自 2019 起） |
| 健保申報延遲與回補 | 最近 1–2 週低報，回測過度樂觀 | 尾端不完整週刪除；共變數平移；建立首報/終報校正 |
| 峰值時間不穩定（第 26、39、44 週都出現過） | h > 4 的不確定性大 | 以分位數呈現；長 horizon 只做規劃用途並明示 |
| `device="mps"` 未經官方驗證 | PyTorch 版在 M1 可能出錯 | 主線用 MLX；PyTorch 版用 CPU（單序列數秒內完成） |
| 第 53 週年份 | 季節相位共變數在 53 週年略有錯位 | 以 `週序/該年總週數` 計算相位 |
| 縣市 × 年齡層過於稀疏 | 零值多、分位數退化 | 只在縣市層或年齡層各自聯合預測，不做 418 條逐條預測 |
| 16 GB 記憶體 | 大批量多變量可能吃緊 | `per_core_batch_size` 4–8；回測分批 |

---

## 10. 參考連結

- GitHub：https://github.com/google-research/timesfm
- Hugging Face 權重：https://huggingface.co/google/timesfm-3.0-pytorch
- Google Research 部落格（TimesFM-3）：https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/
- PyPI：https://pypi.org/project/timesfm/
- 官方 3.0 共變數範例 notebook：`timesfm3-usage/notebooks/promotion_and_synthetic_parity_demo.ipynb`
- 2.5 LoRA 微調範例：`timesfm-forecasting/examples/finetuning/`
- 疾管署資料開放平台（流感）：https://data.cdc.gov.tw/group/flu
- 傳染病統計資料查詢系統：https://nidss.cdc.gov.tw
- 健保門診及住院就診人次統計-類流感（政府資料開放平臺）：https://data.gov.tw/dataset/14593

---

## 11. 實作狀態（2026-09-17）

### 11.1 已完成

| 項目 | 狀態 | 位置 |
|---|---|---|
| 環境 | `.venv`（Python 3.11.6）+ `timesfm[mlx]` 3.0.2、pandas、pyarrow、matplotlib、statsforecast | `pyproject.toml`、`README.md` |
| 資料管線 | cp950 讀取、疫情週對應、頭尾不完整週自動偵測、縣市/年齡層名稱正規化、NIDDS 隱性零值補 0；產出全國/縣市/年齡層面板與 QA 報告（11 秒） | `src/forecast_teller/{io,weeks,panel}.py`、`scripts/build_panel.py`、`data_processed/` |
| 共變數 | 季節相位 sin/cos、春節週旗標、每週平日假日天數（`tw_holiday.csv`，2026 起以農曆新年日期外推） | `src/forecast_teller/covariates.py` |
| 模型包裝 | MLX 後端 `TimesFM3Forecaster`，支援單/多變量、過去與未來共變數、log1p、對稱平均、非負裁切 | `src/forecast_teller/model_timesfm3.py` |
| 回測引擎 | 擴張/滑動視窗、任意國家級目標、多變量聯合目標、過去共變數延遲平移；WIS/MAE/WAPE/MASE/涵蓋率；三段分層 | `src/forecast_teller/{backtest,metrics,baselines}.py`、`scripts/run_backtest.py` |
| 縣市/年齡層回測 | 逐序列單變量 vs 聯合多變量（22 縣市 / 18 年齡層一次 forward） | `scripts/run_group_backtest.py` |
| 報表 | 排行榜、WIS by horizon、各年、偏差/MAPE 診斷、高流行週、圖 | `scripts/make_report.py`、`src/forecast_teller/report.py` |
| 即時預測 | 從各指標最新完整週往後 4 週，輸出分位數 CSV 與扇形圖 | `scripts/forecast_now.py`、`outputs/latest/` |
| 檢查 | 週/共變數/指標/基準的基本檢查 | `tests/test_basics.py` |

### 11.2 資料更新後的實際覆蓋（`data_processed/coverage.json`）

| 來源 | 原始範圍 | 判定完整 | 被移除的週 |
|---|---|---|---|
| 健保門診/住院 | 200501–202608 | 200502–202606 | 200501（1 天）、202607（春節週且申報未齊）、202608 |
| 健保急診 | 201552–202608 | 201601–202607 | 201552、202608 |
| RODS | 200901–202608 | 200901–202607 | 202608（僅 3 天） |
| NIDDS 重症 | 200302–202607 | 200302–202603 | 202604–202607（發病週回補延遲，保守多刪 3 週） |
| 合約實驗室 | 201453–202603 | 201502–202602 | 201453、201501、202603 |

回測窗 2016w01–2025w53 內五個來源皆無缺週，共 **522 週**；暖機 104 週後，起點為 2018w01–2025w50 的目標週，共 **415 個起點**。

### 11.3 實作中發現並修正的問題

- `tw_holiday.csv` 在 2021 與 2023–2025 年把節日名稱放在 `holidayCategory` 而非 `name`，最初的春節旗標漏掉這些年份；已改為同時比對兩欄（`tests/test_basics.py` 有檢查）。
- 2026 年以後假日檔尚無資料，春節週的平日假日天數以 5 天外推；請用官方辦公日曆表延伸 `tw_holiday.csv` 後重跑。
- 資料檔擷取日為 2026-02-24，健保 202607 週（2026 春節週）申報明顯未齊，自動偵測已將其排除；即時預測以 202606 為起點。

## 12. 回測結果（2026-09-17，資料窗 2016w01–2025w53，415 個起點，h = 1–4）

完整表格與圖：`outputs/backtest/nhi_out_ili_REPORT.md`、`rods_ili_pct_REPORT.md`、`county_nhi_out_ili_REPORT.md`、`age_nhi_out_ili_REPORT.md`。

### 12.1 全國類流感門診人次：COVID 後（2023–2025）h = 1–4 平均

| 設定 | WIS | MAE | MASE | 80% 涵蓋 | 相對季節性 naive |
|---|---:|---:|---:|---:|---:|
| **tfm_best_mv4_cny_hol**（門診 + 急診 + RODS + 重症四變量聯合，春節旗標 + 假日天數） | **6,857** | 8,725 | 0.35 | 0.82 | 0.30 |

MAPE（COVID 後 h = 1–4 平均）：最佳設定 10.5%，無共變數 zero-shot 12.2%，AutoETS 13.5%，last-value naive 13.1%，MA3 14.0%，季節性 naive 36.7%（見 `nhi_out_ili_REPORT.md` 排行榜）。

基準模型的分位數 = 點預測 + context 內歷史誤差的經驗分位數，誤差分布以其中位數置中，因此中位數就是規則本身（最後一週值、前 3 週平均、去年同週）；MA3 在所有目標與 horizon 都略差於 last-value naive（相對 naive 1.06–1.13），因為移動平均會落後於上升或下降的趨勢。

| 設定 | WIS | MAE | MASE | 80% 涵蓋 | 相對季節性 naive |
|---|---:|---:|---:|---:|---:|
| tfm_best_mv4_hol（同上，再加季節相位） | 6,886 | 8,829 | 0.35 | 0.82 | 0.30 |
| tfm_best_mv3_hol（門診 + 急診 + RODS 三變量聯合，季節 + 春節 + 假日） | 6,910 | 8,854 | 0.35 | 0.80 | 0.30 |
| tfm_cov_cny_hol（**最佳單變量**：擴張視窗、春節旗標 + 假日天數） | 7,065 | 9,044 | 0.36 | 0.79 | 0.31 |
| tfm_cov_hol（只加假日天數） | 7,109 | 9,037 | 0.36 | 0.79 | 0.31 |
| tfm_mv_out_er_rods_sev（門診 + 急診 + RODS + 重症四變量聯合，季節 + 春節） | 7,120 | 9,168 | 0.36 | 0.83 | 0.31 |
| tfm_cov_season_cny_hol | 7,230 | 9,295 | 0.37 | 0.81 | 0.32 |
| tfm_po_rods（RODS 急診% 當過去共變數，延遲 1 週） | 7,275 | 9,306 | 0.37 | 0.81 | 0.32 |
| tfm_expanding（無共變數的 zero-shot） | 7,762 | 9,810 | 0.39 | 0.82 | 0.34 |
| AutoETS | 8,861 | 11,064 | 0.44 | 0.79 | 0.39 |
| Theta | 9,026 | 10,943 | 0.44 | 0.77 | 0.40 |
| last-value naive | 9,205 | 10,779 | 0.43 | 0.63 | 0.40 |
| MA3（前 3 週移動平均，2 週以上遞迴代入；2026-09-17 加入） | 9,746 | 11,376 | 0.46 | 0.62 | 0.43 |
| 季節性 naive | 22,812 | 28,803 | 1.16 | 0.65 | 1.00 |

WIS 依 horizon（COVID 後）：最佳設定 4,145 / 5,794 / 7,993 / 9,497，naive 6,123 / 8,150 / 10,379 / 12,132，即 1–4 週分別改善 32% / 29% / 23% / 22%；相對無共變數的 zero-shot（5,262 / 6,758 / 8,761 / 10,268）改善 21% / 14% / 9% / 8%。COVID 前（2018–2019）最佳為 tfm_best_mv4_hol 4,262，對 naive 6,642 改善 36%，對 ETS 6,329 改善 33%。

### 12.2 各層結論

- **Layer 1（zero-shot）**：不加任何東西的 TimesFM 已比 naive 低 16%、比 ETS 低 12%。log1p 轉換、對稱平均、滑動視窗都沒有幫助；近 3 季滑動視窗明顯較差。80% 區間涵蓋率 0.82，校準良好。
- **Layer 2（已知未來共變數）**：**每週平日假日天數是最有用的共變數**（1 週前 WIS 再降 17%），春節旗標再加一點，季節相位 sin/cos 沒有幫助、長 horizon 略差。假日天數捕捉的是春節、清明、端午、中秋、國慶等造成的門診量下降與反彈。
- **Layer 3（過去共變數與多變量）**：RODS 急診%、實驗室 A/B 型、重症當過去共變數只有 3–6% 的小幅增益；門診 + 急診 + RODS + 重症四變量聯合預測在 h = 3–4 最好（9,607 對單變量最佳 9,863），在 h = 1 較差。
- **Layer 4（多變量 + 假日共變數）**：把 Layer 2 最有用的假日/春節共變數加到四變量聯合預測上，在 COVID 後每個 horizon 都是最佳（6,857），COVID 前也僅次於加季節相位的版本；80%/60% 涵蓋率 0.82/0.62，校準最好。滑動 260 週視窗略差。
- **偏差**：COVID 後整體帶號偏差接近零（h1 +1.6%、h4 0.0%）。高流行週（實際值在第 75 百分位以上）低估 2–8%（h1→h4），COVID 期高估 4–12%。這是零樣本模型的典型行為：對峰值略保守。
- **統計基準**：AutoETS 與 naive 相當，Theta 更差；MA3（前 3 週移動平均）比 naive 差 6–13%；季節性 naive 因季節位移與 COVID 斷層而完全失效（MASE > 1）。

### 12.3 其他目標與層級

- **RODS 急診類流感就診百分比**（`rods_ili_pct_REPORT.md`）：COVID 後最佳為 tfm_expanding_sym，WIS 0.63 個百分點，對 naive 0.72 改善 12%，AutoETS 0.77，季節性 naive 2.0；MAPE 6.8%，MASE 0.25，80% 涵蓋率 0.80。以流行閾值 10% 計：閾值命中率 0.92、誤報率 0.26（naive 0.90 / 0.30）。
- **22 縣市**（`county_nhi_out_ili_REPORT.md`）：22 縣市聯合多變量 + 季節/春節共變數的總 WIS 相對逐縣市 naive 為 0.86，逐縣市單變量 TimesFM 為 0.87，聯合但無共變數 0.89；每個縣市都優於 naive（0.79–0.90），離島（澎湖、金門、連江）增益最小。
- **全國類流感急診就診人次（健保）**（`nhi_er_ili_REPORT.md`，2026-09-17 加入，精簡組合 `--suite core`）：COVID 後最佳同樣是四變量聯合 + 春節 + 假日，WIS 656，相對 naive 0.70（1–4 週分別 −27% / −33% / −34% / −33%），MAPE 10.4%，MASE 0.31，80% 涵蓋率 0.81，方向命中率 0.72；無共變數 zero-shot 相對 naive 0.77，AutoETS 0.97。
- **18 年齡層**（`age_nhi_out_ili_REPORT.md`）：COVID 後逐層單變量 TimesFM 與聯合多變量 + 共變數打平（相對 naive 皆 0.83），聯合無共變數 0.85；COVID 前聯合 + 共變數最佳（0.72，單變量 0.78）；全期聯合 + 共變數 0.80。年齡層的跨序列相關比縣市更強，聯合預測在正常季節的效益也更大。
- **Layer 4**：見 12.1 與 12.2；最佳設定 `tfm_best_mv4_cny_hol`。

### 12.4 即時預測（`outputs/latest/`，2026-02-24 擷取的資料）

聯合三變量（門診 + 急診 + RODS，共同最新完整週 202606，春節旗標 + 假日天數；`latest_forecast_joint.csv`）：

| 指標 | 202607（春節週） | 202608 | 202609 | 202610 | 80% 區間（202607） |
|---|---:|---:|---:|---:|---|
| 全國類流感門診人次 | 75,373 | 91,723 | 87,262 | 85,524 | 64,274–88,798 |
| 全國類流感急診人次 | 10,129 | 7,735 | 7,381 | 7,051 | 5,982–15,423 |
| RODS 急診類流感人次 | 22,731 | 12,935 | 12,128 | 11,850 | 15,477–30,775 |

模型從假日共變數推得春節週門診下降、急診暴增再回落的型態。202607 的急診與 RODS 實際值已完整，可直接驗證：全國類流感急診實際 10,944（預測中位數 10,129，落在 80% 區間內）、RODS 急診類流感實際 21,765（預測 22,731，落在區間內）；門診的 202607 申報尚未齊全，待資料更新後再比對。

單變量（春節旗標 + 假日天數；`latest_forecast.csv`）：RODS 急診% 自 202607 起 10.4 → 10.0 → 9.8 → 9.7（第 1 週 80% 區間 9.7–11.3）；重症自 202603 起 19.5 → 21.0 → 22.3 → 24.6（80% 區間 12.0–35.1）。

### 12.5 建議預設設定

- 全部 horizon：`tfm_best_mv4_cny_hol`，即門診 + 急診 + RODS + 重症四變量聯合、擴張視窗、春節旗標 + 假日天數共變數、`make_positive=True`（`scripts/forecast_now.py --joint nhi_out_ili nhi_er_ili rods_ili nidds_severe --covariates cny holiday`）。
- 聯合預測的 context 必須截到四個指標共同的最新完整週；重症（發病週）通常落後 3 週，若不想犧牲門診資料的時效，改用三變量（門診 + 急診 + RODS，`tfm_best_mv3_hol`，WIS 6,910）或單變量 `tfm_cov_cny_hol`（7,065）。
- 每週流程：更新資料 → `build_panel.py` → `forecast_now.py --joint ... --covariates cny holiday`；假日檔需先延伸至次年。
- 尚未做：集成（TimesFM + ETS）、微調、健保申報回補校正、峰值時間/高度的季節指標。

### 12.6 命中率（hit rate，2026-09-17 加入）

定義（都由已存的逐起點預測直接計算，`scripts/add_hit_rates.py`）：

- `dir_hit` 方向命中率：中位數相對起點週（最後觀測週）的漲/跌方向與實際一致的比例。
- `dir_hit3` 三分類命中率：漲 / 持平 / 跌，實際或預測變動在 ±5% 內視為持平（對應 FluSight 的趨勢分類）。
- `hit_tol10` 容忍帶命中率：中位數落在實際值 ±10% 內的比例。`hit80` 區間命中率 = 80% 涵蓋率。
- `thr_hit_rate` 閾值命中率（敏感度）：實際 ≥ 閾值的週中，中位數也 ≥ 閾值的比例；另報誤報率與整體準確率。閾值：RODS 急診類流感就診百分比用流行閾值 **10%**（使用者指定，`backtest.DEFAULT_THRESHOLDS`），其他目標預設為資料窗內實際值第 75 百分位（門診 79,955 人次）；可用 `make_report.py --threshold` 覆寫。
- `mape` 平均絕對百分比誤差（%）與 MASE 並列。
- last-value naive 的中位數等於起點值，方向命中率退化（永不預測上漲、三分類永遠持平），僅供對照。

全國類流感門診人次，COVID 後 h = 1–4 平均：

| 設定 | dir_hit | dir_hit3 | hit_tol10 | hit80 | 閾值命中率 | 誤報率 | 準確率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tfm_best_mv4_cny_hol | **0.68** | **0.49** | **0.63** | 0.82 | 0.85 | 0.17 | 0.84 |
| tfm_cov_cny_hol | 0.67 | 0.48 | 0.60 | 0.79 | 0.85 | 0.16 | 0.84 |
| tfm_expanding（無共變數） | 0.63 | 0.42 | 0.59 | 0.82 | **0.86** | 0.17 | **0.85** |
| AutoETS | 0.42 | 0.34 | 0.54 | 0.79 | 0.81 | 0.17 | 0.82 |
| Theta | 0.43 | 0.32 | 0.56 | 0.77 | 0.82 | 0.18 | 0.82 |
| last-value naive | 0.55 | 0.32 | 0.57 | 0.63 | 0.82 | 0.17 | 0.82 |
| 季節性 naive | 0.51 | 0.38 | 0.22 | 0.65 | 0.37 | 0.20 | 0.58 |

方向命中率依 horizon（最佳設定）：0.70 / 0.71 / 0.66 / 0.65；ETS 與 Theta 在 0.37–0.47，低於丟銅板，因為指數平滑會把趨勢往均值拉回。閾值命中率各 TimesFM 設定在 h = 1–2 約 0.86–0.90，h = 4 約 0.79–0.80，誤報率約 0.17；加共變數版本在假日週會把預測拉低，敏感度略低於無共變數版本，但準確率相同。

其他層級（COVID 後 h = 1–4 平均）：RODS 急診% 最佳設定方向命中率 0.68（naive 0.51）、閾值命中率 0.74（naive 0.66）、準確率 0.79（0.72）；22 縣市聯合 + 共變數 0.64（naive 0.52）；18 年齡層聯合 + 共變數 0.68（naive 0.53）。完整表格在各 `*_REPORT.md` 的「命中率」一節。

## 13. Dashboard 與 GitHub Pages（2026-09-17）

- `docs/index.html` 每週預測：四項指標的 M1 扇形圖（觀測實線、預測中位數虛線、60%/80% 淺綠帶、預測起點與春節週標線；歷史範圍 40 週 / 1.5 年 / 2.5 年切換）、KPI 與變化方向、分位數表、方法與注意事項。
- `docs/backtest.html` 回測：目標序列與分段切換、可排序排行榜（WIS、相對 naive、MASE、涵蓋率、命中率、閾值命中/誤報）、WIS / 方向命中 / 涵蓋率 / 閾值命中依 horizon、最佳設定預測 vs 實際（h 可切換）、各年 WIS、22 縣市與 18 年齡層相對 WIS 橫條與分段表。
- `docs/report.html` 一頁式解讀與評估報告，由 `scripts/build_site.py` 以最新數字填入，可列印。
- 圖表風格依《疫情資料視覺化指引》v1.1：主色 Sage、折線加深版 `#5D7F58`、類別順序綠→藍→黃、基準用中性灰虛線與不同點形狀、紅色不作類別色、Noto Sans/Serif TC、tabular-nums、僅水平格線、直條圖 Y 軸從零。
- 每週更新：`scripts/weekly_update.sh [--push]`；資料流為 `outputs/` → `docs/data/*.json`（`build_site.py`）→ 靜態頁面 fetch。GitHub Pages 設定步驤見 `README.md`。
