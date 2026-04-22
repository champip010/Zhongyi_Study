/**
 * acupoints.js — 穴位大全 (Acupoint Directory) View
 * Full browse/filter/search of all 370 acupoints from AcuKG dataset
 */

const AcupointsApp = {
    data: [],
    filtered: [],
    activeMeridian: 'all',

    meridianMeta: {
        LU: { name: '肺经', fullName: '手太阴肺经', color: '#8B5CF6' },
        LI: { name: '大肠经', fullName: '手阳明大肠经', color: '#D97706' },
        ST: { name: '胃经', fullName: '足阳明胃经', color: '#F59E0B' },
        SP: { name: '脾经', fullName: '足太阴脾经', color: '#10B981' },
        HT: { name: '心经', fullName: '手少阴心经', color: '#EF4444' },
        SI: { name: '小肠经', fullName: '手太阳小肠经', color: '#F97316' },
        BL: { name: '膀胱经', fullName: '足太阳膀胱经', color: '#3B82F6' },
        KI: { name: '肾经', fullName: '足少阴肾经', color: '#6366F1' },
        PC: { name: '心包经', fullName: '手厥阴心包经', color: '#EC4899' },
        TE: { name: '三焦经', fullName: '手少阳三焦经', color: '#14B8A6' },
        GB: { name: '胆经', fullName: '足少阳胆经', color: '#84CC16' },
        LR: { name: '肝经', fullName: '足厥阴肝经', color: '#22C55E' },
        GV: { name: '督脉', fullName: '督脉', color: '#F43F5E' },
        CV: { name: '任脉', fullName: '任脉', color: '#06B6D4' },
    },

    setupEvents() {
        const searchInput = document.getElementById('acu-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => this.filter(e.target.value));
        }
    },

    init() {
        // Get data from AcuKG global  
        if (window.AcuKG && window.AcuKG.dataList) {
            this.data = window.AcuKG.dataList;
        } else if (typeof acuKGData !== 'undefined') {
            this.data = acuKGData;
        } else {
            this.data = [];
        }

        this.filtered = [...this.data];
        this.activeMeridian = 'all';

        // Reset search
        const searchInput = document.getElementById('acu-search');
        if (searchInput) searchInput.value = '';

        this.buildMeridianFilter();
        this.renderList();
    },

    buildMeridianFilter() {
        const container = document.getElementById('meridian-filter');
        if (!container) return;

        const allBtn = `<button class="meridian-btn active" data-meridian="all">全部 (${this.data.length})</button>`;
        
        const meridianBtns = Object.entries(this.meridianMeta).map(([code, meta]) => {
            const count = this.data.filter(a => a.meridianCode === code).length;
            if (count === 0) return '';
            return `<button class="meridian-btn" data-meridian="${code}" style="--m-color: ${meta.color}">${meta.name} (${count})</button>`;
        }).join('');

        container.innerHTML = allBtn + meridianBtns;

        // Attach click events
        container.querySelectorAll('.meridian-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                container.querySelectorAll('.meridian-btn').forEach(b => b.classList.remove('active'));
                e.currentTarget.classList.add('active');
                this.activeMeridian = e.currentTarget.getAttribute('data-meridian');
                this.filter(document.getElementById('acu-search')?.value || '');
            });
        });
    },

    filter(query) {
        query = (query || '').toLowerCase().trim();

        this.filtered = this.data.filter(item => {
            const matchMeridian = this.activeMeridian === 'all' || item.meridianCode === this.activeMeridian;
            if (!matchMeridian) return false;

            if (!query) return true;
            return (
                (item.zh && item.zh.includes(query)) ||
                (item.pinyin && item.pinyin.toLowerCase().includes(query)) ||
                (item.code && item.code.toLowerCase().includes(query)) ||
                (item.en && item.en.toLowerCase().includes(query))
            );
        });

        this.renderList();
    },

    renderList() {
        const container = document.getElementById('acu-list-container');
        if (!container) return;

        if (this.filtered.length === 0) {
            container.innerHTML = `<div class="acu-list-empty"><i class="fa-solid fa-circle-question"></i><p>没有找到相关穴位</p></div>`;
            return;
        }

        const html = this.filtered.map(item => {
            const meta = this.meridianMeta[item.meridianCode] || { name: '经外奇穴', color: '#94A3B8' };
            const indCount = item.indications ? item.indications.length : 0;
            const preview = item.indications ? item.indications.slice(0, 2).join('、') : '';
            const hasLoc = item.location && item.location.length > 0;

            return `
                <div class="acu-card" onclick="window.AcuKG.openAcupointInfo('${item.code}')">
                    <div class="acu-card-top" style="border-color: ${meta.color}">
                        <div class="acu-card-code" style="color: ${meta.color}">${item.code}</div>
                        <div class="acu-card-name">${item.zh || '—'}</div>
                        <div class="acu-card-badge" style="background-color: ${meta.color}22; color: ${meta.color}; border-color: ${meta.color}44">${meta.name}</div>
                    </div>
                    <div class="acu-card-body">
                        <div class="acu-card-pinyin">${item.pinyin || ''}</div>
                        ${hasLoc ? `<div class="acu-card-loc"><i class="fa-solid fa-map-pin"></i> ${item.location[0]}</div>` : ''}
                        ${preview ? `<div class="acu-card-preview">${preview}${indCount > 2 ? ` +${indCount - 2}` : ''}</div>` : ''}
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = html;
    }
};

window.AcupointsApp = AcupointsApp;
