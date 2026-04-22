/**
 * Application Routing & Base Logic
 */

const App = {
    init() {
        this.cacheDOM();
        this.bindEvents();
        this.initTheme();
        this.updateDashboardStats();
    },

    cacheDOM() {
        this.navLinks = document.querySelectorAll('.nav-links a');
        this.views = document.querySelectorAll('.view');
        this.pageTitle = document.getElementById('page-title');
        this.themeBtn = document.getElementById('theme-btn');
        
        // Dashboard Stats
        this.statTotal = document.getElementById('stat-total');
        this.statScore = document.getElementById('stat-score');
    },

    bindEvents() {
        this.navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const viewName = e.currentTarget.getAttribute('data-view');
                this.switchView(viewName);
            });
        });

        this.themeBtn.addEventListener('click', () => {
            this.toggleTheme();
        });
    },

    switchView(viewName) {
        // Update Nav
        this.navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('data-view') === viewName) {
                link.classList.add('active');
                
                // Update title based on nav text
                const text = link.innerText.trim();
                this.pageTitle.innerText = text.split(' ')[0]; // just get the chinese part
            }
        });

        // Update Views
        this.views.forEach(view => {
            if (view.id === `view-${viewName}`) {
                view.classList.remove('hidden');
            } else {
                view.classList.add('hidden');
            }
        });

        // Trigger view specific logic
        this.triggerViewInit(viewName);
    },

    triggerViewInit(viewName) {
        switch (viewName) {
            case 'dashboard':
                this.updateDashboardStats();
                break;
            case 'flashcards':
                if (window.FlashcardApp) window.FlashcardApp.init();
                break;
            case 'quiz':
                if (window.QuizApp) window.QuizApp.init();
                break;
            case 'manage':
                if (window.ManageApp) window.ManageApp.init();
                break;
            case 'acupoints':
                if (window.AcupointsApp) window.AcupointsApp.init();
                break;
            case 'neike':
                if (window.NeikeApp) window.NeikeApp.init();
                break;
        }
    },

    initTheme() {
        const savedTheme = localStorage.getItem('acustudy_theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        this.updateThemeIcon(savedTheme);
    },

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('acustudy_theme', newTheme);
        this.updateThemeIcon(newTheme);
    },

    updateThemeIcon(theme) {
        if (theme === 'dark') {
            this.themeBtn.innerHTML = '<i class="fa-solid fa-sun"></i>';
        } else {
            this.themeBtn.innerHTML = '<i class="fa-solid fa-moon"></i>';
        }
    },

    updateDashboardStats() {
        const data = Store.getData();
        this.statTotal.innerText = data.length;
        
        const highestScore = localStorage.getItem('acustudy_high_score') || 0;
        this.statScore.innerText = `${highestScore}%`;
    }
};

window.app = App; // Expose to global for HTML onclicks

document.addEventListener('DOMContentLoaded', () => {
    App.init();
    
    // Auto-init view specific modules that don't need strict refresh
    if (window.FlashcardApp) window.FlashcardApp.setupEvents();
    if (window.QuizApp) window.QuizApp.setupEvents();
    if (window.ManageApp) window.ManageApp.setupEvents();
    if (window.AcupointsApp) window.AcupointsApp.setupEvents();
    if (window.NeikeApp) window.NeikeApp.setupEvents();
});
