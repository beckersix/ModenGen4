/**
 * Module Status Tracker
 * Handles fetching and displaying module statuses in the UI
 */

class ModuleStatusTracker {
    constructor() {
        this.modules = {
            'gan': { loaded: false, name: 'GAN System', icon: 'fa-cube', stats: {} },
            'llm': { loaded: false, name: 'Language Processor', icon: 'fa-language', stats: {} },
            'text_to_3d': { loaded: false, name: 'Text to 3D Manager', icon: 'fa-object-group', stats: {} }
        };
        
        this.statusElement = document.getElementById('module-status-container');
        this.fetchInterval = null;
    }
    
    /**
     * Initialize the status tracker
     */
    init() {
        this.fetchStatuses();
        // Refresh status every 30 seconds
        this.fetchInterval = setInterval(() => this.fetchStatuses(), 30000);
    }
    
    /**
     * Fetch module statuses from the API
     */
    fetchStatuses() {
        fetch('/api/module-status/')
            .then(response => response.json())
            .then(data => {
                if (data.modules) {
                    Object.keys(data.modules).forEach(key => {
                        if (this.modules[key]) {
                            this.modules[key].loaded = data.modules[key].loaded;
                            this.modules[key].stats = data.modules[key].stats || {};
                        }
                    });
                    this.updateStatusDisplay();
                }
            })
            .catch(error => {
                console.error('Error fetching module statuses:', error);
            });
    }
    
    /**
     * Update the status indicators in the UI
     */
    updateStatusDisplay() {
        if (!this.statusElement) return;
        
        // Clear existing indicators
        this.statusElement.innerHTML = '';
        
        // Create indicators for each module
        Object.keys(this.modules).forEach(key => {
            const module = this.modules[key];
            const indicator = document.createElement('div');
            indicator.className = 'module-status-indicator';
            
            // Status badge
            const badge = document.createElement('span');
            badge.className = `badge ${module.loaded ? 'bg-success' : 'bg-danger'}`;
            badge.title = `${module.name}: ${module.loaded ? 'Loaded' : 'Not Loaded'}`;
            
            // Icon
            const icon = document.createElement('i');
            icon.className = `fas ${module.icon}`;
            badge.appendChild(icon);
            
            indicator.appendChild(badge);
            
            // Add tooltip with stats if any exist
            if (module.loaded && Object.keys(module.stats).length > 0) {
                indicator.setAttribute('data-bs-toggle', 'tooltip');
                indicator.setAttribute('data-bs-html', 'true');
                
                let tooltipContent = `<strong>${module.name}</strong><br>`;
                Object.keys(module.stats).forEach(stat => {
                    tooltipContent += `${stat}: ${module.stats[stat]}<br>`;
                });
                
                indicator.setAttribute('data-bs-title', tooltipContent);
            }
            
            this.statusElement.appendChild(indicator);
        });
        
        // Initialize tooltips
        if (typeof bootstrap !== 'undefined') {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.moduleStatusTracker = new ModuleStatusTracker();
    window.moduleStatusTracker.init();
});
