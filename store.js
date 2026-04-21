/**
 * App Store - Manages state and LocalStorage for the application
 * Relies on `acupointData` loaded from data.js
 */

const Store = {
    key: 'acupoint_data_v1',
    data: [],

    init() {
        // Check if we have data in LocalStorage
        const savedData = localStorage.getItem(this.key);
        if (savedData) {
            try {
                this.data = JSON.parse(savedData);
                console.log('Loaded data from LocalStorage:', this.data.length);
            } catch (e) {
                console.error("Failed to parse LocalStorage data", e);
                this.loadDefaultData();
            }
        } else {
            this.loadDefaultData();
        }
    },

    loadDefaultData() {
        // Fallback to data.js variable
        if (typeof acupointData !== 'undefined') {
            // Need to generate unique IDs for generic operations
            this.data = acupointData.map((item, index) => ({
                id: `p_${Date.now()}_${index}`,
                disease: item['病名'] || '',
                syndrome: item['证型'] || '',
                manifestations: item['临床表现'] || '',
                principles: item['治疗原则'] || '',
                primary: item['主穴'] || '',
                secondary: item['配穴'] || ''
            }));
            this.save();
            console.log('Loaded default data from data.js');
        } else {
            console.error('acupointData is not available!');
            this.data = [];
        }
    },

    save() {
        localStorage.setItem(this.key, JSON.stringify(this.data));
    },

    getData() {
        return this.data;
    },

    resetToDefault() {
        localStorage.removeItem(this.key);
        this.loadDefaultData();
        return this.data;
    },

    addRecord(record) {
        record.id = `p_${Date.now()}`;
        this.data.unshift(record);
        this.save();
        return record;
    },

    updateRecord(id, updatedFields) {
        const index = this.data.findIndex(r => r.id === id);
        if (index !== -1) {
            this.data[index] = { ...this.data[index], ...updatedFields };
            this.save();
            return this.data[index];
        }
        return null;
    },

    deleteRecord(id) {
        this.data = this.data.filter(r => r.id !== id);
        this.save();
    }
};

// Initialize immediately when script is loaded
Store.init();
