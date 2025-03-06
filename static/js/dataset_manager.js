// Dataset Manager JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Setup CSRF token for AJAX requests
    setupCSRFToken();
    
    // Initialize dataset management functionality
    initializeDatasetManager();

    // Event handlers
    $('#createCustomDatasetForm').on('submit', handleCreateCustomDataset);
    $('#createCombinedDatasetForm').on('submit', handleCreateCombinedDataset);
    $('#download-all-shapenet').on('click', handleDownloadAllShapeNet);
    $('#download-all-objectnet3d').on('click', handleDownloadAllObjectNet3D);
    $('#download-objectnet3d-toolbox').on('click', handleDownloadObjectNet3DToolbox);
    
    // Refresh button for ObjectNet3D
    $('.refresh-objectnet3d').on('click', function() {
        loadObjectNet3DCategories(true);
    });
});

function setupCSRFToken() {
    // Function to get CSRF token from cookies
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    // Add CSRF token to AJAX requests
    const csrftoken = getCookie('csrftoken');
    $.ajaxSetup({
        beforeSend: function(xhr, settings) {
            if (!/^(GET|HEAD|OPTIONS|TRACE)$/i.test(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    });
}

function initializeDatasetManager() {
    // Load ShapeNet categories
    loadShapeNetCategories();

    // Load ObjectNet3D categories
    loadObjectNet3DCategories();

    // Load custom datasets
    loadCustomDatasets();

    // Load combined datasets
    loadCombinedDatasets();
}

function loadShapeNetCategories() {
    console.log('Loading ShapeNet categories...');
    $.ajax({
        url: '/api/datasets/shapenet/',
        method: 'GET',
        success: function(data) {
            console.log('ShapeNet categories loaded:', data);
            renderShapeNetCategories(data);
        },
        error: function(xhr, status, error) {
            console.error('Error loading ShapeNet categories:', error);
            console.error('Response:', xhr.responseText);
            showError('Failed to load ShapeNet categories');
        }
    });
}

function loadObjectNet3DCategories(showLoading = false) {
    if (showLoading) {
        $('#objectnet3d-categories').html(`
            <div class="col-12 text-center">
                <div class="spinner-border" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `);
    }
    
    $.ajax({
        url: '/api/datasets/objectnet3d/categories',
        method: 'GET',
        success: function(data) {
            renderObjectNet3DCategories(data);
        },
        error: function(xhr, status, error) {
            $('#objectnet3d-categories').html(`
                <div class="col-12">
                    <div class="alert alert-danger" role="alert">
                        Error loading ObjectNet3D categories: ${error}
                    </div>
                </div>
            `);
        }
    });
}

function loadCustomDatasets() {
    $.ajax({
        url: '/api/datasets/custom/',
        method: 'GET',
        success: function(data) {
            renderCustomDatasets(data);
        },
        error: function(xhr, status, error) {
            showError('Failed to load custom datasets');
        }
    });
}

function loadCombinedDatasets() {
    $.ajax({
        url: '/api/datasets/combined/',
        method: 'GET',
        success: function(data) {
            renderCombinedDatasets(data);
        },
        error: function(xhr, status, error) {
            showError('Failed to load combined datasets');
        }
    });
}

function handleCreateCustomDataset(e) {
    e.preventDefault();
    // Handle custom dataset creation
}

function handleCreateCombinedDataset(e) {
    e.preventDefault();
    // Handle combined dataset creation
}

function renderShapeNetCategories(data) {
    console.log('Rendering ShapeNet categories:', data);
    const container = $('#shapenet-categories');
    container.empty();

    if (!data.datasets || data.datasets.length === 0) {
        container.html('<div class="col-12"><p>No ShapeNet categories found.</p></div>');
        return;
    }

    data.datasets.forEach(category => {
        console.log('Rendering category:', category);
        const card = `
            <div class="col-md-4 mb-4">
                <div class="card dataset-card">
                    <div class="card-body">
                        <h5 class="card-title">${category.name}</h5>
                        ${category.downloaded ? 
                            `<p class="card-text">${category.model_count} models available</p>` :
                            `<p class="card-text text-muted">Not downloaded</p>`
                        }
                        <div class="dataset-actions">
                            ${!category.downloaded ?
                                `<button class="btn btn-primary btn-sm download-shapenet" data-category-id="${category.id}">
                                    Download
                                </button>` :
                                `<button class="btn btn-success btn-sm" disabled>
                                    Ready
                                </button>`
                            }
                        </div>
                    </div>
                </div>
            </div>`;
        container.append(card);
    });

    // Add click handler for download buttons
    $('.download-shapenet').on('click', handleDownloadShapeNet);
}

function renderObjectNet3DCategories(data) {
    console.log('Rendering ObjectNet3D categories:', data);
    const container = $('#objectnet3d-categories');
    container.empty();

    if (!data || data.length === 0) {
        container.html('<div class="col-12"><p>No ObjectNet3D categories found.</p></div>');
        return;
    }

    data.forEach(category => {
        console.log('Rendering ObjectNet3D category:', category);
        let statusText = '';
        let statusClass = '';
        let buttonClass = '';
        let buttonText = '';
        let buttonDisabled = false;
        
        if (category.downloaded) {
            if (category.is_procedural) {
                statusText = `${category.model_count} procedurally generated models (fallback)`;
                statusClass = 'text-warning';
                buttonClass = 'btn-warning';
                buttonText = 'Procedural';
                buttonDisabled = true;
            } else {
                statusText = `${category.model_count} models available`;
                statusClass = 'text-success';
                buttonClass = 'btn-success';
                buttonText = 'Ready';
                buttonDisabled = true;
            }
        } else {
            statusText = 'Not downloaded';
            statusClass = 'text-muted';
            buttonClass = 'btn-primary';
            buttonText = 'Download';
            buttonDisabled = false;
        }
        
        const card = `
            <div class="col-md-4 mb-4">
                <div class="card dataset-card">
                    <div class="card-body">
                        <h5 class="card-title">${category.name}</h5>
                        <p class="card-text ${statusClass}">${statusText}</p>
                        <div class="dataset-actions">
                            <button class="btn ${buttonClass} btn-sm ${!buttonDisabled ? 'download-objectnet3d' : ''}" 
                                    ${buttonDisabled ? 'disabled' : ''} 
                                    data-category-id="${category.id}">
                                ${buttonText}
                            </button>
                        </div>
                    </div>
                </div>
            </div>`;
        container.append(card);
    });

    // Add click handler for download buttons
    $('.download-objectnet3d').on('click', handleDownloadObjectNet3D);
}

function handleDownloadShapeNet() {
    const button = $(this);
    const categoryId = button.data('category-id');
    button.prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status"></span> Downloading...');

    $.ajax({
        url: '/api/datasets/shapenet/',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({category_id: categoryId}),
        success: function(data) {
            if (data.status === 'success') {
                loadShapeNetCategories();
            } else {
                showError('Failed to download dataset');
                button.prop('disabled', false).text('Download');
            }
        },
        error: function(xhr, status, error) {
            showError('Failed to download dataset');
            button.prop('disabled', false).text('Download');
        }
    });
}

function handleDownloadObjectNet3D() {
    const button = $(this);
    const categoryId = button.data('category-id');
    button.prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status"></span> Downloading...');

    $.ajax({
        url: '/api/datasets/objectnet3d/download',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({category_id: categoryId}),
        success: function(data) {
            if (data.success) {
                if (data.is_procedural) {
                    // Show warning message for procedurally generated fallback models
                    showMessage(`Note: ${data.message || 'Procedurally generated models were created as fallbacks.'}`, 'warning');
                    button.removeClass('btn-primary').addClass('btn-warning').text('Procedural');
                } else {
                    // Show success message for real models
                    showMessage(`Category ${categoryId} downloaded successfully`, 'success');
                    button.removeClass('btn-primary').addClass('btn-success').text('Ready');
                }
                loadObjectNet3DCategories();
            } else {
                showError(data.error || 'Failed to download dataset');
                button.prop('disabled', false).text('Download');
            }
        },
        error: function(xhr, status, error) {
            let errorMessage = 'Failed to download dataset';
            if (xhr.responseJSON && xhr.responseJSON.message) {
                errorMessage = xhr.responseJSON.message;
            }
            showError(errorMessage);
            button.prop('disabled', false).text('Download');
        }
    });
}

function handleDownloadObjectNet3DToolbox() {
    const button = $('#download-objectnet3d-toolbox');
    button.prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status"></span> Downloading toolbox...');

    $.ajax({
        url: '/api/datasets/objectnet3d/download-toolbox',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({}),
        success: function(data) {
            if (data.success) {
                showMessage('ObjectNet3D toolbox downloaded successfully', 'success');
                button.prop('disabled', false).html('<i class="fas fa-tools me-2"></i>Download ObjectNet3D Toolbox');
            } else {
                showError('Failed to download ObjectNet3D toolbox');
                button.prop('disabled', false).html('<i class="fas fa-tools me-2"></i>Download ObjectNet3D Toolbox');
            }
        },
        error: function(xhr, status, error) {
            showError('Failed to download ObjectNet3D toolbox');
            button.prop('disabled', false).html('<i class="fas fa-tools me-2"></i>Download ObjectNet3D Toolbox');
        }
    });
}

function handleDownloadAllShapeNet() {
    const button = $('#download-all-shapenet');
    const progressContainer = $('#shapenet-download-progress');
    const progressBar = progressContainer.find('.progress-bar');
    
    // Disable button and show progress
    button.prop('disabled', true);
    button.html('<span class="spinner-border spinner-border-sm" role="status"></span> Starting download...');
    progressContainer.removeClass('d-none');
    
    $.ajax({
        url: '/api/datasets/shapenet/download-all/',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({}),
        success: function(data) {
            if (data.status === 'success') {
                // Show success message
                showMessage(data.message, 'success');
                
                // Start checking download status periodically
                checkDownloadStatus();
            } else {
                // Show error and reset button
                showError('Failed to start download: ' + data.message);
                resetDownloadButton();
            }
        },
        error: function(xhr, status, error) {
            showError('Failed to start download: ' + error);
            resetDownloadButton();
        }
    });
}

function handleDownloadAllObjectNet3D() {
    const button = $('#download-all-objectnet3d');
    const progressContainer = $('#objectnet3d-download-progress');
    const progressBar = progressContainer.find('.progress-bar');
    
    // Disable button and show progress
    button.prop('disabled', true);
    button.html('<span class="spinner-border spinner-border-sm" role="status"></span> Starting download...');
    progressContainer.removeClass('d-none');
    
    $.ajax({
        url: '/api/datasets/objectnet3d/download-all',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({}),
        success: function(data) {
            if (data.status === 'started') {
                // Start checking download status periodically
                checkObjectNet3DDownloadStatus(data.status_key);
            } else {
                // Show message and reset button
                showMessage(data.message, 'info');
                resetObjectNet3DDownloadButton();
            }
        },
        error: function(xhr, status, error) {
            // Show error and reset button
            showError('Failed to start download: ' + xhr.responseText);
            resetObjectNet3DDownloadButton();
        }
    });
}

function checkDownloadStatus() {
    // Check ShapeNet categories every 5 seconds to update progress
    const progressInterval = setInterval(function() {
        $.ajax({
            url: '/api/datasets/shapenet/',
            method: 'GET',
            success: function(data) {
                updateDownloadProgress(data);
                
                // If all categories are downloaded, stop checking
                const downloadedCount = data.datasets.filter(cat => cat.downloaded).length;
                const totalCount = data.datasets.length;
                
                if (downloadedCount === totalCount) {
                    clearInterval(progressInterval);
                    finishDownload();
                }
            }
        });
    }, 5000);
}

function checkObjectNet3DDownloadStatus(statusKey) {
    $.ajax({
        url: '/api/datasets/objectnet3d/download-status',
        method: 'GET',
        data: {status_key: statusKey},
        success: function(data) {
            updateObjectNet3DDownloadProgress(data);
            
            if (data.status === 'completed') {
                finishObjectNet3DDownload();
            } else {
                // Check again in a second
                setTimeout(function() {
                    checkObjectNet3DDownloadStatus(statusKey);
                }, 1000);
            }
        },
        error: function(xhr, status, error) {
            showError('Error checking download status');
            resetObjectNet3DDownloadButton();
        }
    });
}

function updateDownloadProgress(data) {
    const downloadedCount = data.datasets.filter(cat => cat.downloaded).length;
    const totalCount = data.datasets.length;
    const progressPercent = Math.round((downloadedCount / totalCount) * 100);
    
    // Update progress bar
    const progressBar = $('#shapenet-download-progress .progress-bar');
    progressBar.css('width', progressPercent + '%');
    progressBar.attr('aria-valuenow', progressPercent);
    
    // Update status message
    $('#shapenet-download-progress small').text(
        `Downloading ShapeNet library: ${downloadedCount} of ${totalCount} categories (${progressPercent}%)`
    );
    
    // Refresh the category display
    renderShapeNetCategories(data);
}

function updateObjectNet3DDownloadProgress(data) {
    const progressBar = $('#objectnet3d-download-progress').find('.progress-bar');
    const statusText = $('#objectnet3d-download-progress').find('small');
    
    if (data.percent !== undefined) {
        progressBar.css('width', data.percent + '%');
        progressBar.attr('aria-valuenow', data.percent);
    }
    
    // Show the current component being downloaded
    if (data.completed !== undefined && data.total !== undefined) {
        let currentComponent = "components";
        
        // Identify current component from progress data
        if (data.progress) {
            const components = {
                'cad_models': '3D Models',
                'images': 'Images',
                'annotations': 'Annotations',
                'splits': 'Dataset Splits',
                'toolbox': 'Toolbox'
            };
            
            // Find the current component being processed
            if (data.completed > 0 && data.completed <= data.total) {
                const componentKeys = Object.keys(components);
                if (data.completed <= componentKeys.length) {
                    currentComponent = components[componentKeys[data.completed - 1]];
                }
            }
        }
        
        statusText.html(`Downloading ObjectNet3D dataset (${data.completed}/${data.total}): <strong>${currentComponent}</strong>`);
        
        // Add detailed progress information
        if (data.progress) {
            const progressDetails = $('<div class="mt-2 small"></div>');
            
            Object.keys(data.progress).forEach(component => {
                const componentStatus = data.progress[component];
                let statusHtml = '';
                
                if (componentStatus.status === 'completed') {
                    statusHtml = `<div class="text-success"><i class="fas fa-check-circle"></i> ${component}: Complete</div>`;
                } else if (componentStatus.status === 'error') {
                    statusHtml = `<div class="text-danger"><i class="fas fa-times-circle"></i> ${component}: Error - ${componentStatus.message}</div>`;
                } else {
                    statusHtml = `<div class="text-primary"><i class="fas fa-spinner fa-spin"></i> ${component}: In progress</div>`;
                }
                
                progressDetails.append(statusHtml);
            });
            
            // Check if details already exist and replace them
            const existingDetails = $('#objectnet3d-download-details');
            if (existingDetails.length) {
                existingDetails.html(progressDetails.html());
            } else {
                progressDetails.attr('id', 'objectnet3d-download-details');
                $('#objectnet3d-download-progress').append(progressDetails);
            }
        }
    }
}

function finishDownload() {
    // Update UI when download is complete
    $('#download-all-shapenet').prop('disabled', false).html('<i class="fas fa-check me-2"></i>Download Complete');
    $('#shapenet-download-progress small').text('All ShapeNet categories have been downloaded successfully.');
    
    // Show success message
    showMessage('ShapeNet library has been downloaded successfully!', 'success');
}

function finishObjectNet3DDownload() {
    // Show success message
    showMessage('ObjectNet3D dataset downloaded successfully! The dataset is now ready for use in model training.', 'success');
    
    // Add note about dataset
    const note = `
        <div class="alert alert-info mt-3">
            <h5><i class="fas fa-info-circle"></i> About ObjectNet3D</h5>
            <p>ObjectNet3D is a large scale database for 3D object recognition with 100 categories and 90,127 images.</p>
            <p>The dataset includes 3D CAD models, images, annotations, and a MATLAB toolbox for working with the data.</p>
            <p>Use this dataset for training your 3D generative models with real-world objects and shapes.</p>
            <p><a href="http://cvgl.stanford.edu/projects/objectnet3d/" target="_blank">Learn more about ObjectNet3D <i class="fas fa-external-link-alt"></i></a></p>
        </div>
    `;
    
    // Add the note to the page
    const noteElement = $(note);
    const existingNote = $('#objectnet3d-note');
    if (existingNote.length) {
        existingNote.replaceWith(noteElement);
    } else {
        noteElement.attr('id', 'objectnet3d-note');
        $('#objectnet3d-categories').before(noteElement);
    }
    
    // Reload categories
    loadObjectNet3DCategories();
    
    // Reset UI
    resetObjectNet3DDownloadButton();
}

function resetDownloadButton() {
    // Reset button and hide progress
    $('#download-all-shapenet').prop('disabled', false).html('<i class="fas fa-download me-2"></i>Download Complete ShapeNet Library');
    $('#shapenet-download-progress').addClass('d-none');
}

function resetObjectNet3DDownloadButton() {
    $('#download-all-objectnet3d').prop('disabled', false).html('<i class="fas fa-download me-2"></i>Download Complete ObjectNet3D Library');
    $('#objectnet3d-download-progress').addClass('d-none');
    $('#objectnet3d-download-details').remove();
}

function showError(message) {
    const alert = `
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>`;
    $('main.container-fluid').prepend(alert);
}

function showMessage(message, type = 'info') {
    const alertClass = type === 'success' ? 'alert-success' : 
                      type === 'warning' ? 'alert-warning' : 
                      type === 'danger' ? 'alert-danger' : 'alert-info';
    
    const alertHtml = `
        <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>`;
    
    $('#messages').append(alertHtml);
    
    // Automatically remove after 5 seconds
    setTimeout(() => {
        $('.alert').alert('close');
    }, 5000);
}
