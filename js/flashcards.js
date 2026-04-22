/**
 * Flashcard Feature — Dual-Mode: Illness vs Acupoint
 */

const FlashcardApp = {
    currentIndex: 0,
    deck: [],
    isFlipped: false,
    studyMode: 'illness', // 'illness' | 'acupoint'

    setupEvents() {
        this.container = document.getElementById('fc-container');
        this.progressText = document.getElementById('fc-progress');

        document.getElementById('fc-prev').addEventListener('click', () => this.prevCard());
        document.getElementById('fc-next').addEventListener('click', () => this.nextCard());
        document.getElementById('fc-flip').addEventListener('click', () => this.flipCard());
        document.getElementById('fc-shuffle').addEventListener('click', () => this.shuffleDeck());

        // Mode toggle buttons
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const mode = e.currentTarget.getAttribute('data-mode');
                this.switchMode(mode);
            });
        });
    },

    switchMode(mode) {
        this.studyMode = mode;
        this.currentIndex = 0;

        // Update toggle button appearance
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.getAttribute('data-mode') === mode);
        });

        // Rebuild deck with new data source
        this.loadDeck();
        this.renderCard();
    },

    loadDeck() {
        if (this.studyMode === 'illness') {
            this.deck = [...Store.getData()];
        } else if (this.studyMode === 'acupoint') {
            // Use AcuKG acupoint data
            const acuData = (window.AcuKG && window.AcuKG.dataList) ? window.AcuKG.dataList : [];
            this.deck = acuData.filter(a => a.zh); // only those with a Chinese name
        } else if (this.studyMode === 'neike') {
            // Flatten neikeData patterns into individual cards
            const neike = window.neikeData || [];
            const cards = [];
            neike.forEach(d => {
                d.patterns.forEach(p => {
                    cards.push({
                        ...p,
                        disease: d.disease,
                        category: d.category
                    });
                });
            });
            this.deck = cards;
        }
    },

    init() {
        // Re-set to illness mode and load
        this.studyMode = 'illness';
        this.currentIndex = 0;

        // Ensure toggle UI reflects current mode
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.getAttribute('data-mode') === 'illness');
        });

        this.loadDeck();

        if (this.deck.length === 0) {
            this.container.innerHTML = '<p style="text-align:center; padding: 40px; color: var(--text-muted);">没有找到数据，请检查管理页面。</p>';
            return;
        }

        this.renderCard();
    },

    renderCard() {
        if (this.deck.length === 0) return;

        const cardData = this.deck[this.currentIndex];
        this.isFlipped = false;

        let html;
        if (this.studyMode === 'illness') {
            html = this.renderIllnessCard(cardData);
        } else if (this.studyMode === 'acupoint') {
            html = this.renderAcupointCard(cardData);
        } else if (this.studyMode === 'neike') {
            html = this.renderNeikeCard(cardData);
        }

        this.container.innerHTML = html;
        this.progressText.innerText = `${this.currentIndex + 1} / ${this.deck.length}`;

        // Card click to flip
        const cardElem = document.getElementById('current-card');
        if (cardElem) {
            cardElem.addEventListener('click', (e) => {
                // Don't flip if clicking on an acupoint tooltip trigger
                if (e.target.classList.contains('acu-tooltip-trigger')) return;
                this.flipCard();
            });
        }
    },

    renderIllnessCard(cardData) {
        const primaryHtml = window.AcuKG
            ? window.AcuKG.wrapAcupoints(cardData.primary || 'N/A')
            : (cardData.primary || 'N/A');
        const secondaryHtml = window.AcuKG
            ? window.AcuKG.wrapAcupoints(cardData.secondary || '无')
            : (cardData.secondary || '无');

        return `
            <div class="flashcard" id="current-card">
                <div class="fc-face fc-front">
                    <div class="fc-mode-badge illness-badge">疾病配穴</div>
                    <div class="fc-disease">${cardData.disease}</div>
                    <div class="fc-syndrome">${cardData.syndrome || ''}</div>
                    <div class="fc-manifestation">${cardData.manifestations || ''}</div>
                    <div class="fc-hint"><i class="fa-solid fa-hand-pointer"></i> 点击卡片查看处方</div>
                </div>
                <div class="fc-face fc-back">
                    <h3><i class="fa-solid fa-scale-balanced"></i> 治疗原则</h3>
                    <p>${cardData.principles || 'N/A'}</p>

                    <h3><i class="fa-solid fa-circle-dot"></i> 主穴 (Primary Acupoints)</h3>
                    <p class="acu-names">${primaryHtml}</p>

                    <h3><i class="fa-regular fa-circle-dot"></i> 配穴 (Secondary Acupoints)</h3>
                    <p class="acu-names">${secondaryHtml}</p>

                    <div class="fc-tip"><i class="fa-solid fa-lightbulb"></i> 鼠标悬停穴位名可查看详情，点击可进入穴位百科</div>
                </div>
            </div>
        `;
    },

    renderAcupointCard(acu) {
        const indicationsHtml = (acu.indications && acu.indications.length > 0)
            ? acu.indications.map(i => `<span class="ind-tag">${i}</span>`).join('')
            : '<span style="color:var(--text-muted)">暂无数据</span>';

        // Show meridian prefix as colored badge
        const meridianMap = {
            LU: { name: '肺经', color: '#8B5CF6' }, LI: { name: '大肠经', color: '#D97706' },
            ST: { name: '胃经', color: '#F59E0B' }, SP: { name: '脾经', color: '#10B981' },
            HE: { name: '心经', color: '#EF4444' }, HT: { name: '心经', color: '#EF4444' },
            SI: { name: '小肠经', color: '#F97316' }, BL: { name: '膀胱经', color: '#3B82F6' },
            KI: { name: '肾经', color: '#6366F1' }, PC: { name: '心包经', color: '#EC4899' },
            TE: { name: '三焦经', color: '#14B8A6' }, GB: { name: '胆经', color: '#84CC16' },
            LR: { name: '肝经', color: '#22C55E' }, GV: { name: '督脉', color: '#F43F5E' },
            CV: { name: '任脉', color: '#06B6D4' },
        };
        const prefix = acu.code.replace(/[0-9]/g, '');
        const meridian = meridianMap[prefix] || { name: '经外奇穴', color: '#94A3B8' };

        return `
            <div class="flashcard acu-mode-card" id="current-card">
                <div class="fc-face fc-front acu-card-front">
                    <div class="fc-mode-badge acu-badge">穴位百科</div>
                    <div class="acu-front-code">${acu.code}</div>
                    <div class="acu-front-name">${acu.zh}</div>
                    <div class="acu-front-meridian" style="background-color: ${meridian.color}20; color: ${meridian.color}; border: 1px solid ${meridian.color}40;">${meridian.name}</div>
                    <div class="fc-hint"><i class="fa-solid fa-hand-pointer"></i> 点击查看穴位详情</div>
                </div>
                <div class="fc-face fc-back acu-card-back">
                    <div class="acu-back-header">
                        <span class="acu-back-code">${acu.code}</span>
                        <span class="acu-back-name">${acu.zh}</span>
                        <span class="acu-back-pinyin">${acu.pinyin || ''}</span>
                    </div>
                    <p class="acu-back-en">${acu.en || ''}</p>
                    <h3 style="margin: 16px 0 10px;"><i class="fa-solid fa-notes-medical"></i> 主治适应症</h3>
                    <div class="ind-tags-container">${indicationsHtml}</div>
                </div>
            </div>
        `;
    },
    
    renderNeikeCard(cardData) {
        const herbsHtml = window.AcuKG
            ? window.AcuKG.wrapAcupoints(cardData.herbs || 'N/A')
            : (cardData.herbs || 'N/A');

        return `
            <div class="flashcard neike-mode-card" id="current-card">
                <div class="fc-face fc-front">
                    <div class="fc-mode-badge neike-badge">内科方剂</div>
                    <div class="fc-disease">${cardData.disease}</div>
                    <div class="fc-syndrome">${cardData.name}</div>
                    <div class="fc-manifestation">${cardData.symptoms || ''}</div>
                    <div class="fc-hint"><i class="fa-solid fa-hand-pointer"></i> 点击查看治法与处方</div>
                </div>
                <div class="fc-face fc-back">
                    <div class="neike-back-header">
                        <span class="neike-cat-badge">${cardData.category}</span>
                        <h3>${cardData.disease} · ${cardData.name}</h3>
                    </div>
                    
                    <div class="neike-back-content">
                        <div class="detail-section">
                            <strong>治法：</strong><span>${cardData.principle}</span>
                        </div>
                        <div class="detail-section">
                            <strong>处方：</strong><span style="font-size: 1.2rem; font-weight: 700;">${cardData.formula}</span>
                        </div>
                        <div class="detail-section">
                            <strong>组成：</strong><p class="herbs-text" style="margin-top: 5px;">${herbsHtml}</p>
                        </div>
                        <div class="detail-section">
                            <strong>舌脉：</strong><span>${cardData.tongue} ${cardData.pulse}</span>
                        </div>
                        
                        ${cardData.song ? `
                            <div class="formula-song" style="margin-top: 15px;">
                                <i class="fa-solid fa-music"></i>
                                <p>${cardData.song}</p>
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    },

    flipCard() {
        const cardElem = document.getElementById('current-card');
        if (!cardElem) return;

        this.isFlipped = !this.isFlipped;
        cardElem.classList.toggle('is-flipped', this.isFlipped);
    },

    nextCard() {
        if (this.currentIndex < this.deck.length - 1) {
            this.currentIndex++;
            this.renderCard();
        }
    },

    prevCard() {
        if (this.currentIndex > 0) {
            this.currentIndex--;
            this.renderCard();
        }
    },

    shuffleDeck() {
        for (let i = this.deck.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [this.deck[i], this.deck[j]] = [this.deck[j], this.deck[i]];
        }
        this.currentIndex = 0;
        this.renderCard();
    }
};

window.FlashcardApp = FlashcardApp;
