// acukg_handler.js — Full AcuKG integration: Tooltips, Info Page, Acupoint list
class AcuKGHandler {
  constructor(dataList) {
    this.dataList = dataList || [];
    this.nameToData = {};
    this.codeToData = {};
    this.sortedNames = [];
    this.regex = null;
    this.init();
    this.setupTooltipDOM();
    this.attachEventListeners();
  }

  init() {
    if (!this.dataList.length && typeof acuKGData !== 'undefined') {
      this.dataList = acuKGData;
    }
    this.dataList.forEach(item => {
      this.codeToData[item.code] = item;
      if (item.zh) {
        this.nameToData[item.zh] = item;
        this.sortedNames.push(item.zh);
      }
    });
    this.sortedNames.sort((a, b) => b.length - a.length);
    if (this.sortedNames.length > 0) {
      const escapedNames = this.sortedNames.map(n => n.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
      this.regex = new RegExp(`(${escapedNames.join('|')})`, 'g');
    }
  }

  wrapAcupoints(text) {
    if (!text || typeof text !== 'string') return text;
    if (!this.regex) return text;
    if (text.includes('acu-tooltip-trigger')) return text;
    return text.replace(this.regex, (match) => {
      const data = this.nameToData[match];
      if (data) {
        return `<span class="acu-tooltip-trigger" data-acu-code="${data.code}">${match}</span>`;
      }
      return match;
    });
  }

  setupTooltipDOM() {
    let tooltip = document.getElementById('acu-tooltip');
    if (!tooltip) {
      tooltip = document.createElement('div');
      tooltip.id = 'acu-tooltip';
      tooltip.className = 'acu-tooltip-box';
      tooltip.style.display = 'none';
      document.body.appendChild(tooltip);
    }
    this.tooltipEl = tooltip;
  }

  attachEventListeners() {
    document.addEventListener('mouseover', (e) => {
      if (e.target && e.target.classList.contains('acu-tooltip-trigger')) {
        this.showTooltip(e.target);
      }
    });
    document.addEventListener('mouseout', (e) => {
      if (e.target && e.target.classList.contains('acu-tooltip-trigger')) {
        this.hideTooltip();
      }
    });
    document.addEventListener('click', (e) => {
      if (e.target && e.target.classList.contains('acu-tooltip-trigger')) {
        e.preventDefault();
        e.stopPropagation();
        this.openAcupointInfo(e.target.getAttribute('data-acu-code'));
      }
    });
  }

  showTooltip(targetEl) {
    const code = targetEl.getAttribute('data-acu-code');
    const data = this.codeToData[code];
    if (!data) return;

    const sneakPeek = data.indications && data.indications.length > 0
      ? data.indications.slice(0, 2).join('、')
      : '';

    this.tooltipEl.innerHTML = `
      <div class="tt-header">
        <span class="tt-code">${data.code}</span>
        <span class="tt-name">${data.zh}</span>
      </div>
      <div class="tt-pinyin">${data.pinyin || ''} · ${data.meridian || ''}</div>
      ${sneakPeek ? `<div class="tt-ind"><i class="fa-solid fa-notes-medical"></i> ${sneakPeek}${data.indications.length > 2 ? '...' : ''}</div>` : ''}
      <div class="tt-click-hint">点击查看详情 ›</div>
    `;
    this.tooltipEl.style.display = 'block';

    const rect = targetEl.getBoundingClientRect();
    this.tooltipEl.style.top = '0px';
    this.tooltipEl.style.left = '0px';
    const tooltipRect = this.tooltipEl.getBoundingClientRect();

    let top = rect.top - tooltipRect.height - 10 + window.scrollY;
    let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2) + window.scrollX;
    if (top < window.scrollY) top = rect.bottom + 10 + window.scrollY;
    if (left < 4) left = 4;
    if (left + tooltipRect.width > document.body.clientWidth) {
      left = document.body.clientWidth - tooltipRect.width - 4;
    }
    this.tooltipEl.style.top = `${top}px`;
    this.tooltipEl.style.left = `${left}px`;
  }

  hideTooltip() {
    this.tooltipEl.style.display = 'none';
  }

  openAcupointInfo(code) {
    const data = this.codeToData[code];
    if (!data) return;
    if (window.app && window.app.switchView) {
      window.app.switchView('acupoint-info');
    }
    this.renderInfoPage(data);
  }

  renderInfoPage(data) {
    const container = document.getElementById('acupoint-info-content');
    if (!container) return;

    const meridianColors = {
      LU: '#8B5CF6', LI: '#D97706', ST: '#F59E0B', SP: '#10B981',
      HT: '#EF4444', SI: '#F97316', BL: '#3B82F6', KI: '#6366F1',
      PC: '#EC4899', TE: '#14B8A6', GB: '#84CC16', LR: '#22C55E',
      GV: '#F43F5E', CV: '#06B6D4',
    };
    const mc = data.meridianCode || '';
    const color = meridianColors[mc] || '#94A3B8';

    // Render indications as tags
    const indTags = (data.indications || [])
      .map(i => `<span class="ind-tag">${i}</span>`)
      .join('');

    // Location info from part_of
    const locLines = (data.location || [])
      .slice(0, 5)
      .map(l => `<li>${l}</li>`)
      .join('');

    // Nearby anatomy  
    const nearTags = (data.nearAnatomy || [])
      .map(a => `<span class="anatomy-tag">${a}</span>`)
      .join('');

    // Nearby acupoints
    const nearAcuHtml = (data.nearAcupoints || [])
      .map(c => {
        const d = this.codeToData[c.toUpperCase()] || this.codeToData[c];
        return d ? `<button class="nearby-acu-btn secondary-btn" onclick="window.AcuKG.openAcupointInfo('${d.code}')">${d.zh || d.code}<span>${d.code}</span></button>` : '';
      })
      .filter(Boolean)
      .join('');

    container.innerHTML = `
      <div class="acu-info-header" style="background: linear-gradient(135deg, ${color} 0%, ${color}cc 100%);">
        <div class="acu-info-title-row">
          <h2>${data.zh || data.code}</h2>
          <span class="acu-info-code">${data.code}</span>
          <span class="acu-info-meridian-badge" style="background:rgba(255,255,255,0.2);">${data.meridian || ''}</span>
        </div>
        <p class="acu-info-pinyin">${data.pinyin || ''}</p>
      </div>

      ${data.location && data.location.length > 0 ? `
      <div class="acu-info-section">
        <h3><i class="fa-solid fa-map-pin"></i> 定位</h3>
        <ul>${locLines}</ul>
      </div>` : ''}

      ${data.nearAnatomy && data.nearAnatomy.length > 0 ? `
      <div class="acu-info-section">
        <h3><i class="fa-solid fa-bone"></i> 附近解剖标志</h3>
        <div class="anatomy-tags">${nearTags}</div>
      </div>` : ''}

      ${data.indications && data.indications.length > 0 ? `
      <div class="acu-info-section">
        <h3><i class="fa-solid fa-notes-medical"></i> 主治适应症 <span style="font-size:12px; font-weight:400; color:var(--text-muted);">(${data.indications.length}项)</span></h3>
        <div class="ind-tags-container">${indTags}</div>
      </div>` : ''}

      ${nearAcuHtml ? `
      <div class="acu-info-section">
        <h3><i class="fa-solid fa-circle-nodes"></i> 相邻穴位</h3>
        <div class="nearby-acu-list">${nearAcuHtml}</div>
      </div>` : ''}

      <div class="acu-info-footer">
        <button class="secondary-btn" onclick="history.back(); app.switchView('manage');"><i class="fa-solid fa-arrow-left"></i> 返回</button>
      </div>
    `;
  }
}

// Instantiate globally
window.addEventListener('DOMContentLoaded', () => {
  window.AcuKG = new AcuKGHandler();
});
