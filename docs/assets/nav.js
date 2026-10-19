/* 網站導覽（唯一來源）：上層選疾病，下層選該疾病的頁面。
   每頁只放 <header class="masthead" data-disease="covid" data-page="forecast"></header>，
   緊接著同步載入本檔；網址以本檔位置推得網站根目錄，本機與 GitHub Pages 都適用。 */
(function () {
  const SITE = [
    {key: 'flu', name: '類流感', en: 'INFLUENZA-LIKE ILLNESS', path: 'flu/', pages: [
      ['forecast', '預測', 'index.html', 'WEEKLY FORECAST'], ['backtest', '回測', 'backtest.html', 'BACKTEST'],
      ['report', '報告', 'report.html', 'EVALUATION REPORT'], ['method', '方法', 'method.html', 'METHOD']]},
    {key: 'covid', name: '新冠', en: 'COVID-19', path: 'covid/', pages: [
      ['forecast', '預測', 'index.html', 'WEEKLY FORECAST'], ['backtest', '回測', 'backtest.html', 'BACKTEST'],
      ['report', '報告', 'report.html', 'EVALUATION REPORT'], ['method', '方法', 'method.html', 'METHOD']]},
    {key: 'ev', name: '腸病毒', en: 'ENTEROVIRUS', path: 'ev/', pages: [
      ['forecast', '預測', 'index.html', 'WEEKLY FORECAST'], ['backtest', '回測', 'backtest.html', 'BACKTEST'],
      ['report', '報告', 'report.html', 'EVALUATION REPORT'], ['method', '方法', 'method.html', 'METHOD']]},
    {key: 'dengue', name: '登革熱', en: 'DENGUE', path: 'dengue/', pages: [
      ['alert', '預警', 'index.html', 'DAILY EARLY WARNING'], ['report', '驗證報告', 'report.html', 'VALIDATION REPORT'],
      ['method', '方法', 'method.html', 'METHOD']]},
  ];
  window.FT_SITE = SITE;
  const me = document.currentScript;
  const root = me ? me.src.replace(/assets\/nav\.js.*$/, '') : '/';
  const hdr = document.querySelector('header.masthead[data-page]');
  if (!hdr) return;
  const dkey = hdr.dataset.disease || '', pkey = hdr.dataset.page || '';
  const dz = SITE.find(d => d.key === dkey);
  const pg = dz && dz.pages.find(p => p[0] === pkey);
  const sub = dz ? `${dz.en} · ${pg ? pg[3] : ''}` : 'OVERVIEW';
  const tabs = [`<a href="${root}index.html"${dz ? '' : ' class="active" aria-current="page"'}>總覽</a>`]
    .concat(SITE.map(d => `<a href="${root}${d.path}"${d === dz ? ' class="active"' : ''}>${d.name}</a>`)).join('');
  hdr.innerHTML = `<div class="masthead-inner">
    <a class="masthead-left" href="${root}index.html"><div class="crest">FT</div><div class="masthead-title">台灣重點傳染病預測 · TimesFM 3.0<small>FORECAST-TELLER · ${sub}</small></div></a>
    <nav class="masthead-right" aria-label="疾病">${tabs}</nav></div>`;
  if (dz) {
    const nav = document.createElement('nav');
    nav.className = 'subnav'; nav.setAttribute('aria-label', dz.name + '頁面');
    nav.innerHTML = `<div class="subnav-inner"><span class="subnav-title">${dz.name}</span>` +
      dz.pages.map(p => `<a href="${root}${dz.path}${p[2]}"${p === pg ? ' class="active" aria-current="page"' : ''}>${p[1]}</a>`).join('') + '</div>';
    hdr.after(nav);
  }
})();
