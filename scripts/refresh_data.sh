#!/usr/bin/env bash
# Download the latest weekly surveillance files from Taiwan CDC open data (od.cdc.gov.tw).
# The downloaded files use the open-data schema (possibly UTF-8); the pipeline in
# src/forecast_teller/io.py expects the NHI_PROCESSED_* / RODS_RS layout in data/, so
# map columns before replacing those files.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
curl -sSLo data/raw/NHI_Influenza_like_illness.csv  https://od.cdc.gov.tw/eic/NHI_Influenza_like_illness.csv
curl -sSLo data/raw/NHI_Influenza.csv               https://od.cdc.gov.tw/eic/NHI_Influenza.csv
curl -sSLo data/raw/RODS_Influenza_like_illness.csv https://od.cdc.gov.tw/eic/RODS_Influenza_like_illness.csv
ls -la data/raw/
echo "流感併發重症與實驗室型別資料：https://data.cdc.gov.tw/group/flu 、 https://nidss.cdc.gov.tw"
