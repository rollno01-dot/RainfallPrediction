// Common JavaScript functions for Rainfall Prediction System

// Utility Functions
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
    });
}

function formatNumber(num, decimals = 2) {
    return parseFloat(num).toFixed(decimals);
}

function getConfidenceColor(confidence) {
    const map = {
        'Very High': '#28a745',
        'High': '#007bff',
        'Good': '#17a2b8',
        'Moderate': '#ffc107',
        'Low': '#dc3545',
        'Very Low': '#dc3545'
    };
    return map[confidence] || '#6c757d';
}

function getConfidenceBadge(confidence) {
    const map = {
        'Very High': 'success',
        'High': 'primary',
        'Good': 'info',
        'Moderate': 'warning',
        'Low': 'danger',
        'Very Low': 'danger'
    };
    return map[confidence] || 'secondary';
}

function showLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="text-center py-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-3">Loading data...</p>
            </div>
        `;
    }
}

function hideLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '';
    }
}

function showError(containerId, message) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="alert alert-danger alert-dismissible fade show" role="alert">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
    }
}

function showSuccess(containerId, message) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="alert alert-success alert-dismissible fade show" role="alert">
                <i class="fas fa-check-circle me-2"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
    }
}

// Chart Theme
const chartColors = {
    blue: 'rgba(54, 162, 235, 1)',
    blueLight: 'rgba(54, 162, 235, 0.2)',
    red: 'rgba(255, 99, 132, 1)',
    redLight: 'rgba(255, 99, 132, 0.2)',
    green: 'rgba(75, 192, 192, 1)',
    greenLight: 'rgba(75, 192, 192, 0.2)',
    yellow: 'rgba(255, 206, 86, 1)',
    yellowLight: 'rgba(255, 206, 86, 0.2)',
    purple: 'rgba(153, 102, 255, 1)',
    purpleLight: 'rgba(153, 102, 255, 0.2)',
    orange: 'rgba(255, 159, 64, 1)',
    orangeLight: 'rgba(255, 159, 64, 0.2)'
};

// Chart defaults
Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
Chart.defaults.font.size = 12;

// Export functions
window.formatDate = formatDate;
window.formatNumber = formatNumber;
window.getConfidenceColor = getConfidenceColor;
window.getConfidenceBadge = getConfidenceBadge;
window.showLoading = showLoading;
window.hideLoading = hideLoading;
window.showError = showError;
window.showSuccess = showSuccess;
window.chartColors = chartColors;

// Auto-initialize tooltips
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
});