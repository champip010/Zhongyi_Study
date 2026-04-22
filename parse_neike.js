const fs = require('fs');

const content = fs.readFileSync('pdf_extracted.txt', 'utf8');
const lines = content.split('\n');

const neikeData = [];
let currentCategory = '';
let currentDisease = '';
let currentMainSymptom = '';
let currentPattern = null;

const categories = ['肺', '心脑', '脾', '肝', '肾', '气血津液', '其他'];

// Helper to check if a line is a category
const isCategory = (line) => categories.includes(line.trim());

// Basic parsing logic based on the observed structure
for (let i = 0; i < lines.length; i++) {
    let line = lines[i].trim();
    if (!line) continue;

    if (isCategory(line)) {
        currentCategory = line;
        continue;
    }

    // Check for disease and main symptom (usually followed by "辨证分型")
    if (line.includes('（主症：')) {
        const parts = line.split('（主症：');
        currentDisease = parts[0].trim();
        currentMainSymptom = parts[1].replace('）', '').trim();
        
        neikeData.push({
            category: currentCategory,
            disease: currentDisease,
            mainSymptom: currentMainSymptom,
            patterns: []
        });
        continue;
    }

    // Look for pattern numbers (1., 2., etc.)
    if (/^\d+\.$/.test(line)) {
        const patternName = lines[i+1]?.trim();
        if (patternName) {
            currentPattern = {
                name: patternName,
                symptoms: '',
                principle: '',
                formula: '',
                herbs: '',
                tongue: '',
                song: ''
            };
            
            const currentDiseaseObj = neikeData[neikeData.length - 1];
            if (currentDiseaseObj) {
                currentDiseaseObj.patterns.push(currentPattern);
            }
            i++; // Skip pattern name line
        }
        continue;
    }

    // Basic heuristic to fill in pattern details
    if (currentPattern) {
        if (line.includes('舌脉')) {
            currentPattern.tongue = lines[i+1]?.trim() + (lines[i+2]?.includes('脉') ? ' ' + lines[i+2].trim() : '');
            // i += 1; // Basic skip
        } else if (line.match(/[，。、]{2,}/) || line.length > 15) {
             // Likely symptoms or herbs
             if (!currentPattern.symptoms) currentPattern.symptoms = line;
             else if (!currentPattern.herbs) currentPattern.herbs = line;
        } else if (line.match(/汤|散|丸|饮/)) {
            currentPattern.formula = line;
        } else if (line.match(/解表|清热|化痰|补/)) {
            currentPattern.principle = line;
        }
    }
}

fs.writeFileSync('neike_data_raw.json', JSON.stringify(neikeData, null, 2));
console.log('Parsed diseases:', neikeData.map(d => d.disease).join(', '));
