const fs = require('fs');
const readline = require('readline');
const path = require('path');

const baseDir = path.join(__dirname, 'AcuKG-Knowledge-graph-for-medical-acupuncture', 'AcuKG_json');

// Helper to parse JSON line by line
async function readJsonl(filepath, callback) {
    const fileStream = fs.createReadStream(filepath);
    const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });
    for await (const line of rl) {
        if (!line.trim()) continue;
        try {
            const obj = JSON.parse(line);
            callback(obj);
        } catch(e) {}
    }
}

// Function to extract simplified Chinese from mixed string
// Ex: "雲(云)門(门)" -> "云门"
// Ex: "中府" -> "中府"
function extractSimplified(str) {
    let result = '';
    let i = 0;
    while (i < str.length) {
        if (str[i] === '(') {
            let start = i + 1;
            let end = str.indexOf(')', start);
            if (end !== -1) {
                // If there are multiple chars like (a,b), take first
                let content = str.substring(start, end).split(',')[0];
                // replace the previous character with the simplified inside parenthesi
                result = result.slice(0, -1) + content;
                i = end + 1;
                continue;
            }
        }
        result += str[i];
        i++;
    }
    return result;
}

async function main() {
    const dict = {};

    // 1. Chinese Names
    await readJsonl(path.join(baseDir, 'Chinesename.json'), (row) => {
        const code = row.Acupoint_Code;
        if (!dict[code]) dict[code] = { code, indications: [] };
        // Store both the original and extracted simplified name
        dict[code].zh_original = row.Chinese_Name;
        dict[code].zh = extractSimplified(row.Chinese_Name).trim();
    });

    // 2. English Names
    await readJsonl(path.join(baseDir, 'Englishname.json'), (row) => {
        const code = row.Acupoint_Code;
        if (dict[code]) dict[code].en = row.English_Name.trim();
    });

    // 3. Pinyin Names
    await readJsonl(path.join(baseDir, 'pinyinname.json'), (row) => {
        const code = row.Acupoint_Code;
        if (dict[code]) dict[code].pinyin = row.Pinyin_Name.trim();
    });

    // 4. Indications
    await readJsonl(path.join(baseDir, 'Indication.json'), (row) => {
        const code = row.Acupoint_Code;
        if (dict[code] && row.Indication) {
            dict[code].indications.push(row.Indication.trim());
        }
    });

    // Filter to list and export
    const dataList = Object.values(dict);
    fs.writeFileSync('acukg.js', `const acuKGData = ${JSON.stringify(dataList, null, 2)};\n`);
    console.log(`Extracted ${dataList.length} acupoints. Saved to acukg.js`);
}

main();
