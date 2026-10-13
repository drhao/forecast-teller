#!/usr/bin/env bash
# Download the COVID-19 open-data files from od.cdc.gov.tw into covid_forecast/data/.
# The server omits the TWCA intermediate certificate, so curl needs a CA bundle that includes it:
# data/twca_intermediate.pem (from http://sslserver.twca.com.tw/cacert/Cyber_SSL_2023.crt) + certifi.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-../.venv/bin/python}"
BUNDLE="$(mktemp)"; cat data/twca_intermediate.pem "$("$PY" -c 'import certifi;print(certifi.where())')" > "$BUNDLE"
curl -sSL --cacert "$BUNDLE" -o data/NHI_COVID-19.csv  https://od.cdc.gov.tw/eic/NHI_COVID-19.csv
curl -sSL --cacert "$BUNDLE" -o data/RODS_COVID-19.csv https://od.cdc.gov.tw/eic/RODS_COVID-19.csv
rm -f "$BUNDLE"
ls -la data/NHI_COVID-19.csv data/RODS_COVID-19.csv
