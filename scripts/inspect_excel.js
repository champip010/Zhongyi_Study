const XLSX = require('xlsx');
const fs = require('fs');

function printFile(filename) {
    console.log(`\n--- ${filename} ---`);
    try {
        const workbook = XLSX.readFile(filename);
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const data = XLSX.utils.sheet_to_json(worksheet);
        console.log(`Sheet name: ${sheetName}`);
        console.log(`Total rows: ${data.length}`);
        console.log('First 5 rows:');
        console.log(JSON.stringify(data.slice(0, 5), null, 2));
    } catch (e) {
        console.error(`Error reading ${filename}:`, e);
    }
}

printFile('../data/สรุปฝังเข็มรักษาminisek.xlsx');
printFile('../data/针灸题库_备份.xlsx');
