/**
 * Quiz Feature
 */

const QuizApp = {
    questions: [],
    currentQIndex: 0,
    score: 0,
    maxQuestions: 10,
    isActive: false,

    setupEvents() {
        this.setupEl = document.getElementById('quiz-setup');
        this.activeEl = document.getElementById('quiz-active');
        this.resultEl = document.getElementById('quiz-result');
        
        this.qNumEl = document.getElementById('quiz-question-num');
        this.qTextEl = document.getElementById('quiz-question-text');
        this.scoreEl = document.getElementById('quiz-current-score');
        this.optionsEl = document.getElementById('quiz-options');
        this.feedbackEl = document.getElementById('quiz-feedback');
        this.nextBtn = document.getElementById('quiz-next');
        
        document.getElementById('quiz-start').addEventListener('click', () => this.startQuiz());
        document.getElementById('quiz-restart').addEventListener('click', () => this.startQuiz());
        this.nextBtn.addEventListener('click', () => this.nextQuestion());
    },

    init() {
        if (!this.isActive) {
            this.setupEl.classList.remove('hidden');
            this.activeEl.classList.add('hidden');
            this.resultEl.classList.add('hidden');
        }
    },

    startQuiz() {
        const quizType = document.querySelector('input[name="quiz-type"]:checked').value;
        
        if (quizType === 'acupuncture') {
            const data = Store.getData();
            if (data.length < 4) {
                alert('数据不足，无法生成测试。请至少添加4个病症。');
                return;
            }
            this.generateQuestions(data);
        } else {
            const neike = window.neikeData || [];
            if (neike.length < 2) {
                alert('内科数据加载失败。');
                return;
            }
            this.generateNeikeQuestions(neike);
        }
        
        this.currentQIndex = 0;
        this.score = 0;
        this.isActive = true;
        
        this.setupEl.classList.add('hidden');
        this.resultEl.classList.add('hidden');
        this.activeEl.classList.remove('hidden');
        
        this.renderQuestion();
    },

    generateQuestions(data) {
        this.questions = [];
        // Shuffle data to pick random diseases
        let shuffledData = [...data].sort(() => 0.5 - Math.random());
        
        const numToGenerate = Math.min(this.maxQuestions, shuffledData.length);
        
        for (let i = 0; i < numToGenerate; i++) {
            const correctItem = shuffledData[i];
            
            // Generate 3 wrong options
            const wrongOptions = [];
            while (wrongOptions.length < 3) {
                const randomItem = data[Math.floor(Math.random() * data.length)];
                if (randomItem.id !== correctItem.id && randomItem.primary !== correctItem.primary && !wrongOptions.includes(randomItem.primary)) {
                    wrongOptions.push(randomItem.primary);
                }
            }

            const options = [...wrongOptions, correctItem.primary].sort(() => 0.5 - Math.random());
            const correctIndex = options.indexOf(correctItem.primary);

            this.questions.push({
                questionText: `下列哪项是 <strong>${correctItem.disease}</strong> (${correctItem.syndrome || '常见'}) 的主穴？`,
                options: options,
                correctIndex: correctIndex,
                explanation: `治疗原则: ${correctItem.principles || 'N/A'}`
            });
        }
    },

    generateNeikeQuestions(data) {
        this.questions = [];
        // Flatten all patterns
        const allPatterns = [];
        data.forEach(d => {
            d.patterns.forEach(p => {
                allPatterns.push({ ...p, disease: d.disease });
            });
        });

        let shuffled = [...allPatterns].sort(() => 0.5 - Math.random());
        const num = Math.min(this.maxQuestions, shuffled.length);

        for (let i = 0; i < num; i++) {
            const correct = shuffled[i];
            const type = Math.random() > 0.5 ? 'formula' : 'principle'; // Randomize question type
            
            const wrongOptions = [];
            while (wrongOptions.length < 3) {
                const rand = allPatterns[Math.floor(Math.random() * allPatterns.length)];
                const val = type === 'formula' ? rand.formula : rand.principle;
                const correctVal = type === 'formula' ? correct.formula : correct.principle;
                
                if (val !== correctVal && !wrongOptions.includes(val)) {
                    wrongOptions.push(val);
                }
            }

            const correctVal = type === 'formula' ? correct.formula : correct.principle;
            const options = [...wrongOptions, correctVal].sort(() => 0.5 - Math.random());
            const correctIndex = options.indexOf(correctVal);

            const qText = type === 'formula' 
                ? `<strong>${correct.disease} · ${correct.name}</strong> 的代表方剂是？`
                : `<strong>${correct.disease} · ${correct.name}</strong> 的治疗原则是？`;

            this.questions.push({
                questionText: qText,
                options: options,
                correctIndex: correctIndex,
                explanation: `症状: ${correct.symptoms.substring(0, 50)}...`
            });
        }
    },

    renderQuestion() {
        const q = this.questions[this.currentQIndex];
        
        this.qNumEl.innerText = `问题 ${this.currentQIndex + 1} / ${this.questions.length}`;
        this.scoreEl.innerText = this.score;
        this.qTextEl.innerHTML = q.questionText;
        
        this.feedbackEl.classList.add('hidden');
        this.nextBtn.classList.add('hidden');
        
        this.optionsEl.innerHTML = '';
        q.options.forEach((optText, index) => {
            const btn = document.createElement('button');
            btn.className = 'quiz-option';
            btn.innerHTML = window.AcuKG ? window.AcuKG.wrapAcupoints(optText) : optText;
            btn.addEventListener('click', () => this.handleAnswer(index));
            this.optionsEl.appendChild(btn);
        });
    },

    handleAnswer(selectedIndex) {
        // Disable options
        const optionBtns = this.optionsEl.querySelectorAll('.quiz-option');
        optionBtns.forEach(btn => btn.disabled = true);

        const q = this.questions[this.currentQIndex];
        const isCorrect = selectedIndex === q.correctIndex;

        if (isCorrect) {
            this.score++;
            this.scoreEl.innerText = this.score;
            optionBtns[selectedIndex].classList.add('correct');
            
            this.feedbackEl.className = 'quiz-feedback success';
            this.feedbackEl.innerHTML = `<i class="fa-solid fa-check-circle"></i> 回答正确！<br><span style="font-size: 14px; opacity: 0.9;">${q.explanation}</span>`;
        } else {
            optionBtns[selectedIndex].classList.add('incorrect');
            optionBtns[q.correctIndex].classList.add('correct');
            
            this.feedbackEl.className = 'quiz-feedback error';
            this.feedbackEl.innerHTML = `<i class="fa-solid fa-times-circle"></i> 回答错误。<br><span style="font-size: 14px; opacity: 0.9;">正确答案是: ${q.options[q.correctIndex]}<br>${q.explanation}</span>`;
        }

        this.feedbackEl.classList.remove('hidden');
        this.nextBtn.classList.remove('hidden');
    },

    nextQuestion() {
        if (this.currentQIndex < this.questions.length - 1) {
            this.currentQIndex++;
            this.renderQuestion();
        } else {
            this.finishQuiz();
        }
    },

    finishQuiz() {
        this.isActive = false;
        this.activeEl.classList.add('hidden');
        this.resultEl.classList.remove('hidden');
        
        const percentage = Math.round((this.score / this.questions.length) * 100);
        document.getElementById('quiz-final-score').innerText = `${percentage}%`;
        
        let text = '再接再厉！';
        if (percentage >= 90) text = '太棒了！针灸大师！';
        else if (percentage >= 70) text = '做得很不错！';
        
        document.getElementById('quiz-final-text').innerText = text;
        
        // Save high score
        const currentHigh = parseInt(localStorage.getItem('acustudy_high_score') || 0);
        if (percentage > currentHigh) {
            localStorage.setItem('acustudy_high_score', percentage);
        }
    }
};

window.QuizApp = QuizApp;
