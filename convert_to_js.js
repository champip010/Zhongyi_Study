const XLSX = require('xlsx');
const fs = require('fs');

try {
    const workbook = XLSX.readFile('针灸题库_备份.xlsx');
    const sheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[sheetName];
    const data = XLSX.utils.sheet_to_json(worksheet);
    
    // Convert to a module that can be loaded in browser
    const fileContent = `const acupointData = ${JSON.stringify(data, null, 2)};\n`;
    fs.writeFileSync('data.js', fileContent);
    console.log(`Saved ${data.length} records to data.js`);
} catch (e) {
    console.error(`Error processing file:`, e);
}
