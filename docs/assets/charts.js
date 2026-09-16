/* 圖表模組：Chart.js 4 + 《疫情資料視覺化指引》規範（M1 不確定性帶、Pattern A/B、加深版折線色） */
(function () {
  const P = {50:'#F6F9F6',100:'#E8EEE7',200:'#D1DECF',300:'#B4C9B1',400:'#91B08C',500:'#739A6D',600:'#5D7F58',700:'#496345',800:'#374C34',900:'#253423'};
  const N = {50:'#FAFAFA',100:'#F2F3F1',200:'#E4E7E4',300:'#CACFC9',400:'#A2ABA0',500:'#7A8778',600:'#5D675B',700:'#444C43',800:'#2C312B',900:'#181B18'};
  const LINE = {primary:'#5D7F58', blue:'#587A9D', yellow:'#A8821F', teal:'#356B70', bronze:'#765A39', plum:'#7A4E5C'};
  const BAND80 = 'rgba(180,201,177,0.22)', BAND60 = 'rgba(180,201,177,0.42)';
  const ACCENT = {alert:'#BE373C', caution:'#D2962D'};
  const FONT = "'Noto Sans TC','Noto Sans',sans-serif";

  Chart.defaults.font.family = FONT;
  Chart.defaults.font.size = 11.5;
  Chart.defaults.color = N[700];
  Chart.defaults.borderColor = N[200];
  Chart.defaults.plugins.legend.position = 'top';
  Chart.defaults.plugins.legend.align = 'start';
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.legend.labels.boxHeight = 8;
  Chart.defaults.plugins.legend.labels.padding = 14;
  Chart.defaults.plugins.tooltip.backgroundColor = N[800];
  Chart.defaults.plugins.tooltip.titleFont = {family: FONT, size: 12, weight: '600'};
  Chart.defaults.plugins.tooltip.bodyFont = {family: FONT, size: 12};
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.boxWidth = 8;
  Chart.defaults.plugins.tooltip.boxHeight = 8;
  Chart.defaults.plugins.tooltip.usePointStyle = true;
  Chart.defaults.elements.line.tension = 0.3;
  Chart.defaults.elements.point.radius = 0;
  Chart.defaults.elements.point.hoverRadius = 4;
  Chart.defaults.animation = false;

  // 垂直/水平參考線 + 標註（M1 規則 5：預測起點）
  const refLines = {
    id: 'refLines',
    afterDatasetsDraw(chart) {
      const o = chart.options.plugins.refLines; if (!o) return;
      const {ctx, chartArea: {top, bottom, left, right}, scales} = chart;
      (o.vertical || []).forEach(l => {
        const x = scales.x.getPixelForValue(l.at); if (!isFinite(x)) return;
        ctx.save(); ctx.strokeStyle = l.color || N[400]; ctx.setLineDash(l.dash || [4, 4]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
        if (l.label) { const w = ctx.measureText(l.label).width; const align = l.align || ((x + 8 + w > right) ? 'right' : 'left'); ctx.fillStyle = l.color || N[600]; ctx.font = `11px ${FONT}`; ctx.textAlign = align; ctx.fillText(l.label, x + (align === 'right' ? -6 : 6), top + 12 + (l.row || 0) * 14); }
        ctx.restore();
      });
      (o.horizontal || []).forEach(l => {
        const y = scales.y.getPixelForValue(l.at); if (!isFinite(y)) return;
        ctx.save(); ctx.strokeStyle = l.color || N[400]; ctx.setLineDash(l.dash || [4, 4]); ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
        if (l.label) { ctx.fillStyle = l.color || N[600]; ctx.font = `11px ${FONT}`; ctx.textAlign = 'right'; ctx.fillText(l.label, right - 4, y - 5); }
        ctx.restore();
      });
    }
  };
  Chart.register(refLines);


  const fmtNum = (v, d = 0) => (v == null || isNaN(v)) ? '—' : Number(v).toLocaleString('zh-TW', {minimumFractionDigits: d, maximumFractionDigits: d});
  const dateTick = (labels) => (val, idx, ticks) => { const s = labels[val]; if (!s) return ''; return s.slice(0, 7).replace('-', '/'); };
  const legendFilter = (item) => !String(item.text).startsWith('_');

  /** M1 扇形圖：觀測實線 + 預測中位數虛線 + 80%/60% 淺色帶 + 預測起點標線 */
  function fanChart(canvas, d, opt = {}) {
    const dec = opt.decimals ?? 0; const unit = opt.unit || '';
    const hist = opt.historyWeeks ? d.history.slice(-opt.historyWeeks) : d.history;
    const labels = hist.map(h => h.date).concat(d.forecast.filter(f => !hist.some(h => h.yw === f.yw)).map(f => f.date));
    const idx = Object.fromEntries(labels.map((l, i) => [l, i]));
    const n = labels.length;
    const arr = () => new Array(n).fill(null);
    const obs = arr(), med = arr(), lo80 = arr(), hi80 = arr(), lo60 = arr(), hi60 = arr();
    hist.forEach(h => { obs[idx[h.date]] = h.y; });
    const originIdx = idx[hist.find(h => h.yw === d.origin_yw)?.date] ?? (hist.length - 1);
    // 讓預測線與帶從最後觀測點接續
    med[originIdx] = lo80[originIdx] = hi80[originIdx] = lo60[originIdx] = hi60[originIdx] = d.last_observed;
    d.forecast.forEach(f => { const i = idx[f.date]; med[i] = f.median; lo80[i] = f.q10; hi80[i] = f.q90; lo60[i] = f.q20; hi60[i] = f.q80; });
    const lastObsIdx = obs.map((v, i) => v != null ? i : -1).filter(i => i >= 0).pop();
    const pointR = obs.map((v, i) => (i === lastObsIdx || i === originIdx) ? 3.5 : 0);
    const fc = d.forecast;
    const cnyIdx = fc.filter(f => f.cny).map(f => idx[f.date]);
    const datasets = [
      {label: '觀測值', data: obs, borderColor: LINE.primary, borderWidth: 2.5, pointRadius: pointR, pointBackgroundColor: LINE.primary, spanGaps: false, order: 1},
      {label: '預測中位數', data: med, borderColor: LINE.primary, borderWidth: 2.5, borderDash: [6, 3], pointRadius: 0, order: 2},
      {label: '80% 預測區間', data: hi80, borderColor: 'transparent', backgroundColor: BAND80, fill: '+1', pointRadius: 0, order: 5, borderWidth: 0},
      {label: '_lo80', data: lo80, borderColor: 'transparent', backgroundColor: 'transparent', pointRadius: 0, order: 5, borderWidth: 0},
      {label: '60% 預測區間', data: hi60, borderColor: 'transparent', backgroundColor: BAND60, fill: '+1', pointRadius: 0, order: 4, borderWidth: 0},
      {label: '_lo60', data: lo60, borderColor: 'transparent', backgroundColor: 'transparent', pointRadius: 0, order: 4, borderWidth: 0},
    ];
    const vertical = [{at: originIdx, label: '預測起點 ' + d.origin_date, color: N[500]}];
    cnyIdx.forEach(i => vertical.push({at: i, label: '春節週', color: '#A8821F', dash: [2, 3], row: 1}));
    return new Chart(canvas, {
      type: 'line', data: {labels, datasets},
      options: {
        responsive: true, maintainAspectRatio: false, interaction: {mode: 'index', intersect: false},
        plugins: {
          legend: {labels: {filter: legendFilter, generateLabels: chart => {
            const items = Chart.defaults.plugins.legend.labels.generateLabels(chart).filter(legendFilter);
            items.forEach(it => { if (it.text.includes('預測區間')) { it.pointStyle = 'rect'; it.fillStyle = it.text.startsWith('80') ? 'rgba(180,201,177,0.5)' : 'rgba(180,201,177,0.9)'; it.strokeStyle = 'transparent'; } else { it.pointStyle = 'line'; } });
            return items; }}},
          refLines: {vertical},
          tooltip: {filter: item => !String(item.dataset.label).startsWith('_') && item.parsed.y != null,
            callbacks: {
              title: items => items.length ? labels[items[0].dataIndex] + (fc.find(f => idx[f.date] === items[0].dataIndex) ? '（預測）' : '') : '',
              label: item => `${item.dataset.label}：${fmtNum(item.parsed.y, dec)} ${unit}`,
              afterBody: items => { const i = items[0]?.dataIndex; const f = fc.find(x => idx[x.date] === i); if (!f) return [];
                const rows = [`q10–q90：${fmtNum(f.q10, dec)} – ${fmtNum(f.q90, dec)}`, `q20–q80：${fmtNum(f.q20, dec)} – ${fmtNum(f.q80, dec)}`];
                if (f.observed != null) rows.push(`實際（已完整）：${fmtNum(f.observed, dec)} ${unit}`);
                if (f.cny) rows.push('春節週'); return rows; }
            }}
        },
        scales: {
          x: {grid: {display: false}, ticks: {maxTicksLimit: 9, maxRotation: 0, callback: dateTick(labels)}, border: {color: N[300]}},
          y: {beginAtZero: true, grid: {color: N[200]}, border: {display: false}, ticks: {callback: v => fmtNum(v, dec)}, title: {display: !!unit, text: unit, font: {size: 11}, color: N[500]}}
        }
      }
    });
  }

  /** 多序列折線（Pattern B：≤3 條類別色 + 中性灰對照；不同點形狀輔助色盲） */
  function linesChart(canvas, series, opt = {}) {
    const colors = [LINE.primary, LINE.blue, LINE.yellow];
    const shapes = ['circle', 'rect', 'triangle'];
    const refShapes = ['rectRot', 'cross', 'crossRot', 'star'];
    let ci = 0, ri = 0;
    const datasets = series.map(s => {
      const isRef = s.kind === 'baseline';
      const color = isRef ? N[400] : (s.color || colors[Math.min(ci, 2)]);
      const style = isRef ? refShapes[(ri++) % refShapes.length] : shapes[Math.min(ci, 2)];
      if (!isRef) ci++;
      return {label: s.label, data: s.data, borderColor: color, backgroundColor: color, borderWidth: isRef ? 1.5 : (ci === 1 ? 2.5 : 2),
              borderDash: isRef ? [5, 4] : [], pointStyle: style, pointRadius: 4, pointHoverRadius: 6, tension: 0.15};
    });
    const fmt = opt.fmt || (v => fmtNum(v, 2));
    return new Chart(canvas, {
      type: 'line', data: {labels: opt.labels || ['1', '2', '3', '4'], datasets},
      options: {responsive: true, maintainAspectRatio: false, interaction: {mode: 'index', intersect: false},
        plugins: {refLines: opt.refLines || {}, tooltip: {callbacks: {title: it => `${opt.xLabel || 'h'} = ${it[0].label}`, label: it => `${it.dataset.label}：${fmt(it.parsed.y)}`}}},
        scales: {x: {grid: {display: false}, title: {display: !!opt.xLabel, text: opt.xLabel, color: N[500], font: {size: 11}}, border: {color: N[300]}},
                 y: {beginAtZero: opt.yMin == null, min: opt.yMin, max: opt.yMax, grid: {color: N[200]}, border: {display: false}, ticks: {callback: fmt}, title: {display: !!opt.yLabel, text: opt.yLabel, color: N[500], font: {size: 11}}}}}
    });
  }

  /** 由回測 segment 資料組成 linesChart 的序列 */
  function byHorizonSeries(seg, cfgs, key, labels) {
    const baselines = new Set(['snaive', 'naive', 'ma3', 'ets', 'theta']);
    return cfgs.filter(c => seg[key][c]).map(c => ({label: labels[c] || c, data: seg[key][c], kind: baselines.has(c) ? 'baseline' : 'model'}));
  }

  /** 回測：預測 vs 實際的時間序列（Pattern A：模型為焦點，實際為深灰對照） */
  function backtestTsChart(canvas, rows, opt = {}) {
    const dec = opt.decimals ?? 0, unit = opt.unit || '';
    const labels = rows.map(r => r.date);
    const datasets = [
      {label: '實際值', data: rows.map(r => r.y), borderColor: N[700], borderWidth: 2, pointRadius: 0, order: 1},
      {label: '預測中位數', data: rows.map(r => r.median), borderColor: LINE.primary, borderWidth: 2.5, borderDash: [6, 3], pointRadius: 0, order: 2},
      {label: '80% 預測區間', data: rows.map(r => r.q90), borderColor: 'transparent', backgroundColor: BAND80, fill: '+1', pointRadius: 0, borderWidth: 0, order: 5},
      {label: '_lo', data: rows.map(r => r.q10), borderColor: 'transparent', backgroundColor: 'transparent', pointRadius: 0, borderWidth: 0, order: 5},
    ];
    if (opt.showNaive) datasets.push({label: 'last-value naive', data: rows.map(r => r.naive), borderColor: N[400], borderWidth: 1.5, borderDash: [3, 3], pointRadius: 0, order: 3});
    return new Chart(canvas, {type: 'line', data: {labels, datasets},
      options: {responsive: true, maintainAspectRatio: false, interaction: {mode: 'index', intersect: false},
        plugins: {legend: {labels: {filter: legendFilter, generateLabels: chart => { const items = Chart.defaults.plugins.legend.labels.generateLabels(chart).filter(legendFilter);
            items.forEach(it => { if (it.text.includes('區間')) { it.pointStyle = 'rect'; it.fillStyle = 'rgba(180,201,177,0.6)'; it.strokeStyle = 'transparent'; } else it.pointStyle = 'line'; }); return items; }}},
          tooltip: {filter: it => !String(it.dataset.label).startsWith('_'), callbacks: {label: it => `${it.dataset.label}：${fmtNum(it.parsed.y, dec)} ${unit}`}}},
        scales: {x: {grid: {display: false}, ticks: {maxTicksLimit: 10, maxRotation: 0, callback: dateTick(labels)}, border: {color: N[300]}},
                 y: {beginAtZero: true, grid: {color: N[200]}, border: {display: false}, ticks: {callback: v => fmtNum(v, dec)}}}}
    });
  }

  /** 橫條排名（Pattern A 單一主色；值直接標在條末；1.0 參考線 = naive） */
  function hbarChart(canvas, labels, values, opt = {}) {
    const fmt = opt.fmt || (v => fmtNum(v, 2));
    // 值標在條末端（inline plugin：閉包持有格式化函式，避免被 Chart.js 當成 scriptable option 呼叫）
    const barLabels = {id: 'barLabels', afterDatasetsDraw(chart) {
      const {ctx} = chart; const meta = chart.getDatasetMeta(0);
      ctx.save(); ctx.font = `11px ${FONT}`; ctx.fillStyle = N[700]; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      meta.data.forEach((bar, i) => { const v = values[i]; if (v == null) return; ctx.fillText(fmt(v), bar.x + 5, bar.y); });
      ctx.restore(); }};
    return new Chart(canvas, {type: 'bar', data: {labels, datasets: [{label: opt.label || '', data: values, backgroundColor: P[500], barPercentage: 0.7, categoryPercentage: 0.85, borderRadius: 0}]},
      plugins: [barLabels],
      options: {indexAxis: 'y', responsive: true, maintainAspectRatio: false,
        plugins: {legend: {display: false}, refLines: opt.refAt != null ? {vertical: [{at: opt.refAt, label: opt.refLabel || '', color: N[500], dash: [4, 4], align: 'left'}]} : {},
                  tooltip: {callbacks: {label: it => `${opt.label || ''} ${fmt(it.parsed.x)}`}}},
        scales: {x: {beginAtZero: true, max: opt.xMax, grid: {color: N[200]}, border: {display: false}, ticks: {callback: fmt}, title: {display: !!opt.xLabel, text: opt.xLabel, color: N[500], font: {size: 11}}},
                 y: {grid: {display: false}, border: {color: N[300]}, ticks: {autoSkip: false, font: {size: 11}}}}}
    });
  }

  window.EPI = {P, N, LINE, ACCENT, fmtNum, fanChart, linesChart, byHorizonSeries, backtestTsChart, hbarChart};
})();
