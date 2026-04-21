/**
 * Manage Feature
 */

const ManageApp = {
    setupEvents() {
        this.tbody = document.getElementById('manage-tbody');
        this.searchInput = document.getElementById('manage-search');
        
        this.searchInput.addEventListener('input', (e) => this.filterData(e.target.value));
        
        document.getElementById('manage-reset').addEventListener('click', () => {
            if (confirm("确定要恢复初始数据吗？所有自定义的无保存更改将丢失。")) {
                Store.resetToDefault();
                this.renderTable(Store.getData());
                alert('数据已恢复。');
            }
        });
    },

    init() {
        const data = Store.getData();
        this.renderTable(data);
        this.searchInput.value = '';
    },

    renderTable(data) {
        this.tbody.innerHTML = '';
        
        if (data.length === 0) {
            this.tbody.innerHTML = '<tr><td colspan="5" style="text-align: center;">没有找到数据。</td></tr>';
            return;
        }

        data.forEach(item => {
            const tr = document.createElement('tr');
            
            // Truncate long texts slightly for display
            const truncate = (str, len) => (str && str.length > len) ? str.substring(0, len) + '...' : str;

            const primaryText = truncate(item.primary, 30) || '-';
            const primaryHtml = window.AcuKG ? window.AcuKG.wrapAcupoints(primaryText) : primaryText;

            tr.innerHTML = `
                <td><strong>${item.disease}</strong></td>
                <td>${item.syndrome || '-'}</td>
                <td>${truncate(item.principles, 15) || '-'}</td>
                <td>${primaryHtml}</td>
                <td>
                    <button class="secondary-btn action-btn edit-btn" data-id="${item.id}" title="目前暂不支持界面编辑，可通过数据源修改"><i class="fa-solid fa-pen"></i></button>
                    <button class="secondary-btn action-btn delete-btn" data-id="${item.id}" style="color: var(--color-error); border-color: var(--color-error);"><i class="fa-solid fa-trash"></i></button>
                </td>
            `;

            this.tbody.appendChild(tr);
        });

        // Add event listeners for delete and edit
        this.tbody.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const id = e.currentTarget.getAttribute('data-id');
                if (confirm("确定要删除这条记录吗？")) {
                    Store.deleteRecord(id);
                    this.init(); // Refresh table
                }
            });
        });
        
        this.tbody.querySelectorAll('.edit-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                alert("内联编辑功能在当前版本暂时省略。(仅作演示，可通过Manage重置恢复数据)");
            });
        });
    },

    filterData(query) {
        query = query.toLowerCase().trim();
        if (!query) {
            this.renderTable(Store.getData());
            return;
        }

        const filtered = Store.getData().filter(item => {
            const searchStr = `${item.disease} ${item.syndrome} ${item.primary} ${item.manifestations} ${item.principles}`.toLowerCase();
            return searchStr.includes(query);
        });
        
        this.renderTable(filtered);
    }
};

window.ManageApp = ManageApp;
