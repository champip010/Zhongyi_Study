/**
 * neike.js — Internal Medicine (内科) Study View
 * Handles rendering, filtering, and searching of Internal Medicine data.
 */

const NeikeApp = {
    data: [],
    filtered: [],
    activeCategory: 'all',

    init() {
        this.data = window.neikeData || [];
        this.filtered = [...this.data];
        this.render();
    },

    setupEvents() {
        // Category filtering
        const catContainer = document.getElementById('neike-categories');
        if (catContainer) {
            catContainer.addEventListener('click', (e) => {
                const btn = e.target.closest('.cat-btn');
                if (btn) {
                    document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    this.activeCategory = btn.getAttribute('data-cat');
                    this.filter();
                }
            });
        }

        // Search
        const searchInput = document.getElementById('neike-search');
        if (searchInput) {
            searchInput.addEventListener('input', () => {
                this.filter();
            });
        }
    },

    filter() {
        const query = document.getElementById('neike-search').value.toLowerCase().trim();
        
        this.filtered = this.data.filter(disease => {
            // Category match
            const matchesCat = this.activeCategory === 'all' || disease.category === this.activeCategory;
            if (!matchesCat) return false;

            // Search match
            if (!query) return true;
            
            const diseaseMatch = disease.disease.toLowerCase().includes(query);
            const patternMatch = disease.patterns.some(p => 
                p.name.toLowerCase().includes(query) || 
                p.formula.toLowerCase().includes(query) ||
                p.herbs.toLowerCase().includes(query)
            );
            
            return diseaseMatch || patternMatch;
        });

        this.render();
    },

    render() {
        const container = document.getElementById('neike-list-container');
        if (!container) return;

        if (this.filtered.length === 0) {
            container.innerHTML = '<div class="no-results">没有找到匹配的内科疾病</div>';
            return;
        }

        container.innerHTML = this.filtered.map(disease => `
            <div class="neike-card">
                <div class="neike-card-header">
                    <span class="neike-cat-badge">${disease.category}</span>
                    <h3>${disease.disease}</h3>
                </div>
                <div class="neike-card-body">
                    <p class="main-symptom"><strong>主症：</strong>${disease.mainSymptom}</p>
                    <div class="patterns-list">
                        ${disease.patterns.map(pattern => `
                            <div class="pattern-item">
                                <div class="pattern-header" onclick="this.parentElement.classList.toggle('expanded')">
                                    <span class="pattern-name">${pattern.name}</span>
                                    <span class="pattern-formula">${pattern.formula}</span>
                                    <i class="fa-solid fa-chevron-down"></i>
                                </div>
                                <div class="pattern-details">
                                    <div class="detail-section">
                                        <strong>治法：</strong><span>${pattern.principle}</span>
                                    </div>
                                    <div class="detail-section">
                                        <strong>症状：</strong><span>${pattern.symptoms}</span>
                                    </div>
                                    <div class="detail-section">
                                        <strong>舌脉：</strong><span>${pattern.tongue} ${pattern.pulse}</span>
                                    </div>
                                    <div class="detail-section">
                                        <strong>处方组成：</strong><span class="herbs-text">${pattern.herbs}</span>
                                    </div>
                                    ${pattern.song ? `
                                        <div class="formula-song">
                                            <i class="fa-solid fa-music"></i>
                                            <p>${pattern.song}</p>
                                        </div>
                                    ` : ''}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `).join('');

        // Apply AcuKG tooltips to herb names and symptoms if needed
        if (window.AcuKG && window.AcuKG.wrapAcupoints) {
            document.querySelectorAll('.herbs-text, .main-symptom, .detail-section span').forEach(el => {
                el.innerHTML = window.AcuKG.wrapAcupoints(el.innerHTML);
            });
        }
    }
};

window.NeikeApp = NeikeApp;
