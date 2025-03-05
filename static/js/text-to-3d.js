/**
 * Text to 3D Generator JavaScript
 * Handles UI interaction, 3D model loading, and API communication
 */

// Import Three.js and addons using the proper import syntax
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

// Make THREE available globally but don't override if already set
if (!window.THREE) {
    window.THREE = THREE;
}

// Store global variables and objects
const viewer = {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    grid: null,
    model: null,
    isInitialized: false,
    animation: null,
    isAnimating: false,
    isLoadingModel: false
};

// Configuration
const config = {
    historyLayout: 'list', // 'list', 'grid-medium', 'grid-small'
    apiEndpoints: {
        generate: '/api/text-to-3d/generate/',
        status: '/api/text-to-3d/status/',
        models: '/api/text-to-3d/available-models/'
    }
};

/**
 * Enhanced fetch function with CSRF token handling
 * @param {string} url - The URL to fetch
 * @param {object} options - Fetch options
 * @returns {Promise} - The fetch promise
 */
function _fetch(url, options = {}) {
    console.log(`_fetch called for ${url}`);
    
    // Default options
    const defaultOptions = {
        credentials: 'same-origin',
        headers: {
            'Content-Type': 'application/json',
        }
    };
    
    // Merge default options with provided options
    const mergedOptions = { ...defaultOptions, ...options };
    
    // If it's a POST, PUT, PATCH or DELETE request, add CSRF token
    if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(mergedOptions.method)) {
        const csrfToken = getCSRFToken();
        if (csrfToken) {
            console.log("Adding CSRF token to request");
            mergedOptions.headers['X-CSRFToken'] = csrfToken;
        } else {
            console.warn("No CSRF token found for non-GET request");
        }
    }
    
    console.log("Request options:", JSON.stringify(mergedOptions));
    
    // Make the fetch call with the merged options
    return fetch(url, mergedOptions);
}

/**
 * Initialize the application when the document is ready
 */
let initialized = false;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded");
    
    // Delay slightly to ensure all resources are loaded
    setTimeout(function() {
        if (!initialized) {
            initialized = true;
            console.log("Initializing from DOM content loaded event");
            
            // Initialize Three.js viewport
            initializeViewport();
            
            // Only add default cube if viewport initialized successfully
            if (viewer.scene) {
                // Add default spinning cube to empty scene
                addDefaultCube();
            }
            
            // Initialize UI components and event handlers
            setupEventListeners();
            initializeHistoryPanel();
            loadTrainedModels();
        }
    }, 300); // Give a brief delay to ensure scripts are loaded
});

/**
 * Set up all event listeners
 */
function setupEventListeners() {
    console.log("Setting up event listeners");
    
    // Layout toggle buttons
    document.querySelectorAll('.layout-toggle-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const layout = this.dataset.layout;
            switchHistoryLayout(layout);
            
            // Update active button
            document.querySelectorAll('.layout-toggle-btn').forEach(b => {
                b.classList.remove('active');
            });
            this.classList.add('active');
        });
    });
    
    
    // Set up detail level slider
    const detailSlider = document.getElementById('detail-level');
    if (detailSlider) {
        detailSlider.addEventListener('input', function() {
            document.getElementById('detail-value').textContent = this.value;
        });
    }
    
    // Set up generate button
    const generateBtn = document.getElementById('generate-btn');
    if (generateBtn) {
        console.log("Found generate button, adding click handler");
        generateBtn.addEventListener('click', function(e) {
            console.log("Generate button clicked");
            e.preventDefault();
            generateModel();
        });
    } else {
        console.error("Generate button not found");
    }
    
    // Set up form submission
    const form = document.getElementById('text-to-3d-form');
    if (form) {
        console.log("Found form, adding submit handler");
        form.addEventListener('submit', function(e) {
            console.log("Form submitted");
            e.preventDefault();
            generateModel();
        });
    } else {
        console.error("Form not found");
    }
    
    // Load model buttons - Add event listeners to history items
    setupHistoryItemListeners();
    
    // Viewport control buttons
    document.querySelectorAll('.viewport-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const action = this.getAttribute('title');
            
            if (action === 'Reset Camera') {
                resetCamera();
            } else if (action === 'Toggle Grid') {
                toggleGrid();
            } else if (action === 'Toggle Wireframe') {
                toggleWireframe();
            }
        });
    });
    
    console.log("Event listeners setup complete");
}

/**
 * Setup event listeners for history items
 */
function setupHistoryItemListeners() {
    console.log("Setting up history item listeners");
    
    // Remove any existing click listeners to prevent duplicates
    document.querySelectorAll('.history-item').forEach(item => {
        const clone = item.cloneNode(true);
        item.parentNode.replaceChild(clone, item);
    });
    
    // Add new listeners with debouncing
    document.querySelectorAll('.history-item').forEach(item => {
        console.log(`Found history item with ID: ${item.dataset.id}`);
        
        // Use a click handler with debouncing
        item.addEventListener('click', debounce(function(e) {
            e.preventDefault();
            
            // Check if item is already selected
            if (this.classList.contains('selected')) {
                console.log('Item already selected, ignoring click');
                return;
            }
            
            const modelId = this.dataset.id;
            console.log(`History item clicked: ${modelId}`);
            
            // Highlight the selected item
            document.querySelectorAll('.history-item').forEach(i => {
                i.classList.remove('selected');
            });
            this.classList.add('selected');
            
            // Load the model
            loadModelById(modelId);
        }, 500)); // 500ms debounce
    });
}

/**
 * Simple debounce function to prevent multiple rapid calls
 */
function debounce(func, wait) {
    let timeout;
    return function() {
        const context = this;
        const args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(() => {
            func.apply(context, args);
        }, wait);
    };
}

/**
 * Initialize the 3D viewport
 */
function initializeViewport() {
    console.log("Initializing 3D viewport");
    
    if (viewer.isInitialized) {
        console.log("Viewport already initialized, not reinitializing");
        return;
    }

    // Get the container element
    const container = document.getElementById('model-viewer');
    if (!container) {
        console.error("Model viewer container not found");
        return;
    }
    
    // Mark as initialized immediately to prevent double initialization
    viewer.isInitialized = true;
    
    // Set up THREE.js scene, camera, and renderer
    viewer.scene = new THREE.Scene();
    viewer.scene.background = new THREE.Color(0x2a2a2a);
    
    // Set up camera
    const width = container.clientWidth;
    const height = container.clientHeight;
    viewer.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    viewer.camera.position.z = 5;
    
    // Set up renderer
    viewer.renderer = new THREE.WebGLRenderer({ antialias: true });
    viewer.renderer.setSize(width, height);
    viewer.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(viewer.renderer.domElement);
    
    // Set up controls
    console.log("Setting up OrbitControls");
    viewer.controls = new OrbitControls(viewer.camera, viewer.renderer.domElement);
    viewer.controls.enableDamping = true;
    viewer.controls.dampingFactor = 0.25;
    viewer.controls.enableZoom = true;
    viewer.controls.enablePan = true;
    viewer.controls.enableRotate = true;
    console.log("OrbitControls initialized:", viewer.controls);
    
    // Add lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
    viewer.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1);
    viewer.scene.add(directionalLight);
    
    const backLight = new THREE.DirectionalLight(0xffffff, 0.5);
    backLight.position.set(-1, -1, -1);
    viewer.scene.add(backLight);
    
    // Add grid
    viewer.grid = new THREE.GridHelper(10, 20, 0x555555, 0x333333);
    viewer.scene.add(viewer.grid);
    
    // Handle window resize
    window.addEventListener('resize', function() {
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        viewer.camera.aspect = width / height;
        viewer.camera.updateProjectionMatrix();
        viewer.renderer.setSize(width, height);
    });
    
    console.log('3D viewport initialized successfully');
    
    // Start animation loop if not already running
    if (!viewer.isAnimating) {
        console.log("Starting animation loop from initializeViewport");
        animate();
    }
}

/**
 * Add a default cube to the scene
 */
function addDefaultCube() {
    console.log("Adding default cube to scene");
    
    // Clear the scene first
    clearScene();
    
    // Create a default cube geometry
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshPhongMaterial({
        color: 0x3498db,
        specular: 0x111111,
        shininess: 30
    });
    
    viewer.defaultCube = new THREE.Mesh(geometry, material);
    viewer.defaultCube.visible = true;
    viewer.scene.add(viewer.defaultCube);
    
    console.log("Default cube added to scene:", viewer.defaultCube);
    
    // Set camera position for default cube
    viewer.camera.position.set(3, 3, 3);
    viewer.camera.lookAt(0, 0, 0);
    viewer.controls.update();
    
    // Hide loading indicator if it's showing
    showLoading(false);
}

/**
 * Create progress overlay elements
 */
function createProgressOverlay(container) {
    // Create overlay div
    viewer.progressOverlay = document.createElement('div');
    viewer.progressOverlay.classList.add('viewport-progress-overlay');
    viewer.progressOverlay.style.display = 'none';
    
    // Create progress container
    const progressContainer = document.createElement('div');
    progressContainer.classList.add('progress-container');
    
    // Create status text
    const statusText = document.createElement('div');
    statusText.classList.add('progress-status');
    statusText.textContent = 'Generating Model...';
    progressContainer.appendChild(statusText);
    
    // Create progress bar
    const progressBarContainer = document.createElement('div');
    progressBarContainer.classList.add('progress');
    
    viewer.progressBar = document.createElement('div');
    viewer.progressBar.classList.add('progress-bar');
    viewer.progressBar.style.width = '0%';
    viewer.progressBar.setAttribute('role', 'progressbar');
    viewer.progressBar.setAttribute('aria-valuenow', '0');
    viewer.progressBar.setAttribute('aria-valuemin', '0');
    viewer.progressBar.setAttribute('aria-valuemax', '100');
    
    progressBarContainer.appendChild(viewer.progressBar);
    progressContainer.appendChild(progressBarContainer);
    
    // Add to overlay
    viewer.progressOverlay.appendChild(progressContainer);
    
    // Add to container
    container.appendChild(viewer.progressOverlay);
}

/**
 * Reset camera to default position
 */
function resetCamera() {
    viewer.camera.position.set(4, 4, 4);
    viewer.camera.lookAt(0, 0, 0);
    viewer.controls.update();
}

/**
 * Toggle grid visibility
 */
function toggleGrid() {
    if (viewer.grid) {
        viewer.grid.visible = !viewer.grid.visible;
    }
}

/**
 * Toggle wireframe mode on the current model
 */
function toggleWireframe() {
    if (viewer.model) {
        viewer.model.traverse(function(child) {
            if (child.isMesh) {
                child.material.wireframe = !child.material.wireframe;
            }
        });
    }
}

/**
 * Load a 3D model from a URL
 * @param {string} modelUrl - The URL of the model to load
 */
function loadModel(modelUrl) {
    console.log("Loading model from URL:", modelUrl);
    
    if (!viewer.isInitialized) {
        console.error('Viewport not initialized');
        return;
    }
    
    // Clear all existing models and objects
    clearScene();
    
    // Show loading indicator
    showLoading(true);
    
    // Set a flag to track if we're currently loading a model
    viewer.isLoadingModel = true;
    
    // Load the model using OBJLoader
    const loader = new OBJLoader();
    loader.load(
        modelUrl,
        function(object) {
            console.log("Model loaded successfully");
            
            // Reset loading flag
            viewer.isLoadingModel = false;
            
            // Fix material settings for all meshes in the model
            object.traverse(function(child) {
                if (child.isMesh) {
                    // Create a new material or modify existing one to fix face culling
                    if (child.material) {
                        // Ensure proper side setting for face culling
                        // THREE.FrontSide: only render front faces (default)
                        // THREE.BackSide: only render back faces
                        // THREE.DoubleSide: render both sides
                        child.material.side = THREE.DoubleSide;
                        
                        // Set proper material properties
                        child.material.flatShading = false;  // Smooth shading
                        child.material.needsUpdate = true;
                    }
                }
            });
            
            // Center the model
            const box = new THREE.Box3().setFromObject(object);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            
            // Normalize size
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 2 / maxDim;
            object.scale.set(scale, scale, scale);
            
            // Center at origin
            object.position.x = -center.x * scale;
            object.position.y = -center.y * scale;
            object.position.z = -center.z * scale;
            
            // Add to scene
            viewer.scene.add(object);
            viewer.model = object;
            
            // Reset camera position to see the model
            viewer.camera.position.z = 5;
            viewer.controls.update();
            
            // Hide loading indicator
            showLoading(false);
            
            // Make sure animation is running
            animate();
        },
        function(xhr) {
            console.log((xhr.loaded / xhr.total * 100) + '% loaded');
        },
        function(error) {
            console.error('Error loading model:', error);
            
            // Reset loading flag
            viewer.isLoadingModel = false;
            
            // Hide loading indicator
            showLoading(false);
            
            // Show default cube on error
            addDefaultCube();
            
            // Make sure animation is running
            animate();
            
            alert('Error loading model: ' + error.message);
        }
    );
}

/**
 * Clear all models and objects from the scene
 */
function clearScene() {
    console.log("Clearing scene of all models and objects");
    
    if (!viewer.scene) {
        console.error("Scene not initialized, cannot clear");
        return;
    }
    
    try {
        // Remove any loaded model
        if (viewer.model) {
            viewer.scene.remove(viewer.model);
            viewer.model = null;
        }
        
        // Make default cube invisible if it exists
        if (viewer.defaultCube && viewer.defaultCube.visible) {
            viewer.defaultCube.visible = false;
        }
        
        // Create arrays of objects to keep and remove
        const keepers = [];
        const toRemove = [];
        
        // Find all lights, grid, and essential objects to keep
        viewer.scene.traverse(function(object) {
            if (object instanceof THREE.Light || 
                object === viewer.grid || 
                object === viewer.camera || 
                object === viewer.scene) {
                keepers.push(object);
            } else if (object !== viewer.defaultCube && 
                      object.type !== 'Scene' && 
                      object.type !== 'Camera') {
                toRemove.push(object);
            }
        });
        
        // Remove non-essential objects
        for (let i = 0; i < toRemove.length; i++) {
            viewer.scene.remove(toRemove[i]);
        }
        
        console.log("Scene cleared successfully");
    } catch (error) {
        console.error("Error while clearing scene:", error);
    }
}

/**
 * Load a model by its ID
 */
function loadModelById(modelId) {
    console.log(`Loading model with ID: ${modelId}`);
    
    // Show loading indicator
    showLoading(true);
    
    _fetch(`/api/text-to-3d/model/${modelId}/details/`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Model data received:', data);
            if (data.mesh_url) {
                loadModel(data.mesh_url);
                
                // Update prompt field with the model's prompt if the element exists
                if (data.prompt) {
                    const promptField = document.getElementById('text-prompt');
                    if (promptField) {
                        promptField.value = data.prompt;
                    } else {
                        console.warn('Prompt field not found in the DOM');
                    }
                }
                
                // Highlight the selected history item
                highlightHistoryItem(modelId);
            } else {
                console.error('No mesh URL found for model');
                showLoading(false);
                addDefaultCube(); // Show default cube when no mesh is available
            }
        })
        .catch(error => {
            console.error('Error fetching model details:', error);
            showLoading(false);
            addDefaultCube(); // Show default cube on error
        });
}

/**
 * Highlight the selected history item and remove highlight from others
 */
function highlightHistoryItem(modelId) {
    // Remove highlight from all history items
    const historyItems = document.querySelectorAll('.history-item');
    historyItems.forEach(item => {
        item.classList.remove('selected');
    });
    
    // Add highlight to the selected item
    const selectedItem = document.querySelector(`.history-item[data-id="${modelId}"]`);
    if (selectedItem) {
        selectedItem.classList.add('selected');
    }
}

/**
 * Initialize history panel
 */
function initializeHistoryPanel() {
    // Set default layout
    switchHistoryLayout(config.historyLayout);
    
    // Set active layout button
    document.querySelectorAll('.layout-toggle-btn').forEach(btn => {
        if (btn.dataset.layout === config.historyLayout) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    // Setup event listeners for history items
    setupHistoryItemListeners();
}

/**
 * Switch between different history layouts
 */
function switchHistoryLayout(layout) {
    const historyContainer = document.getElementById('history-container');
    if (!historyContainer) return;
    
    // Remove existing layout classes
    historyContainer.classList.remove('history-list', 'history-grid-medium', 'history-grid-small');
    
    // Add new layout class
    historyContainer.classList.add(`history-${layout}`);
    
    // Update config
    config.historyLayout = layout;
}

/**
 * Show or hide loading indicator
 */
function showLoading(show) {
    const loadingElement = document.getElementById('loading-indicator');
    if (loadingElement) {
        if (show) {
            // When showing the loading indicator, make sure no animation is happening
            if (viewer.animationId) {
                cancelAnimationFrame(viewer.animationId);
                viewer.animationId = null;
            }
            viewer.isAnimating = false;
            loadingElement.style.display = 'flex';
        } else {
            loadingElement.style.display = 'none';
        }
    }
}

/**
 * Show or hide progress overlay with progress value
 */
function showProgress(show, progress = 0) {
    if (viewer.progressOverlay) {
        viewer.progressOverlay.style.display = show ? 'flex' : 'none';
        
        if (viewer.progressBar) {
            viewer.progressBar.style.width = `${progress}%`;
            viewer.progressBar.setAttribute('aria-valuenow', progress);
        }
    }
}

/**
 * Load available trained models for the dropdown
 */
function loadTrainedModels() {
    const modelSelect = document.getElementById('trained-model');
    if (!modelSelect) return;
    
    console.log("Fetching available trained models...");
    
    _fetch(config.apiEndpoints.models)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            console.log("Received response from models API");
            return response.json();
        })
        .then(data => {
            console.log("Received data:", data);
            
            // Clear existing options (except default)
            while (modelSelect.options.length > 1) {
                modelSelect.remove(1);
            }
            
            // Add new options
            if (Array.isArray(data)) {
                console.log(`Adding ${data.length} models to dropdown`);
                // Filter to only show completed models
                const completedModels = data.filter(model => model.status === 'completed');
                completedModels.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.id;
                    option.textContent = model.name;
                    modelSelect.appendChild(option);
                });
                console.log(`Added ${completedModels.length} completed models to dropdown`);
            } else if (data.models && Array.isArray(data.models)) {
                console.log(`Adding ${data.models.length} models from models array`);
                const completedModels = data.models.filter(model => model.status === 'completed');
                completedModels.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.id;
                    option.textContent = model.name;
                    modelSelect.appendChild(option);
                });
                console.log(`Added ${completedModels.length} completed models to dropdown`);
            }
        })
        .catch(error => {
            console.error('Error loading trained models:', error);
        });
}

/**
 * Get CSRF token from cookies
 */
function getCSRFToken() {
    console.log("Getting CSRF token");
    const name = 'csrftoken';
    let cookieValue = null;
    
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                console.log("Found CSRF token in cookie");
                break;
            }
        }
    }
    
    return cookieValue;
}

/**
 * Generate a new 3D model from the form data
 */
function generateModel() {
    console.log("Generate model function called");
    
    const prompt = document.getElementById('text-prompt').value.trim();
    if (!prompt) {
        alert('Please enter a description of your 3D model');
        return;
    }
    
    // Get output formats
    const formatMesh = document.getElementById('format-mesh').checked;
    const formatPointCloud = document.getElementById('format-point-cloud').checked;
    
    // Validate at least one format is selected
    if (!formatMesh && !formatPointCloud) {
        alert('Please select at least one output format');
        return;
    }
    
    // Get the selected trained model ID
    const selectedModelId = document.getElementById('trained-model').value;
    console.log("Selected model ID:", selectedModelId);
    
    // Collect form data
    const formData = {
        prompt: prompt,
        trained_model_id: selectedModelId,
        detail_level: parseInt(document.getElementById('detail-level').value),
        output_formats: {
            mesh: formatMesh,
            point_cloud: formatPointCloud
        }
    };
    
    console.log("Sending form data:", formData);
    
    // Show generation status
    const generationStatus = document.getElementById('generation-status');
    generationStatus.classList.remove('d-none');
    
    document.getElementById('status-title').textContent = 'Starting model generation...';
    document.getElementById('status-message').textContent = 'Sending request to server...';
    
    // Show progress overlay
    viewer.isGenerating = true;
    showProgress(true, 5);
    
    // Use our enhanced _fetch
    _fetch(config.apiEndpoints.generate, {
        method: 'POST',
        body: JSON.stringify(formData)
    })
    .then(response => {
        console.log("Received response:", response.status);
        if (!response.ok) {
            return response.text().then(text => {
                throw new Error(`Server returned ${response.status}: ${text}`);
            });
        }
        return response.json();
    })
    .then(data => {
        console.log("Generation started successfully:", data);
        document.getElementById('status-title').textContent = 'Model generation started';
        document.getElementById('status-message').textContent = 'Your model is being generated. This may take a few moments.';
        
        // Start polling for status
        const modelId = data.model_id || data.id;
        console.log("Polling for model ID:", modelId);
        
        if (modelId) {
            pollModelStatus(modelId);
        } else {
            console.error("No model ID returned from server");
            document.getElementById('status-title').textContent = 'Error starting generation';
            document.getElementById('status-message').textContent = 'The server did not return a model ID. Please try again.';
        }
    })
    .catch(error => {
        console.error('Error generating model:', error);
        document.getElementById('status-title').textContent = 'Error generating model';
        document.getElementById('status-message').textContent = 'There was an error generating your model: ' + error.message;
        
        // Hide progress overlay after a delay
        setTimeout(() => {
            viewer.isGenerating = false;
            showProgress(false);
        }, 3000);
    });
}

/**
 * Poll for model generation status
 * @param {string} modelId - The ID of the model being generated
 */
function pollModelStatus(modelId) {
    if (!modelId) {
        console.error('No model ID provided for status polling');
        document.getElementById('status-title').textContent = 'Error tracking model generation';
        document.getElementById('status-message').textContent = 'Could not track model generation progress. Please check the history panel for your model.';
        
        // Hide progress overlay after a delay
        setTimeout(() => {
            viewer.isGenerating = false;
            showProgress(false);
        }, 3000);
        return;
    }
    
    console.log(`Polling status for model ${modelId}...`);
    
    // Use the correct URL format - model ID in the path, not as a query parameter
    _fetch(`${config.apiEndpoints.status}${modelId}/`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log(`Status for model ${modelId}:`, data);
            
            // Update progress
            let progressPercent = 5; // Default minimum progress
            
            if (data.progress) {
                progressPercent = Math.max(5, Math.min(99, data.progress));
            } else if (data.status === 'processing') {
                progressPercent = 50;
            } else if (data.status === 'completed') {
                progressPercent = 100;
            }
            
            showProgress(true, progressPercent);
            
            // Update status message
            const statusTitle = document.getElementById('status-title');
            const statusMessage = document.getElementById('status-message');
            
            switch (data.status) {
                case 'pending':
                    statusTitle.textContent = 'Model generation queued';
                    statusMessage.textContent = 'Your model is in the queue. Processing will begin shortly.';
                    break;
                case 'processing':
                    statusTitle.textContent = 'Model generation in progress';
                    statusMessage.textContent = 'Your model is being generated. This may take a few minutes.';
                    break;
                case 'completed':
                    statusTitle.textContent = 'Model generation complete';
                    statusMessage.textContent = 'Your model has been generated successfully!';
                    
                    // Set a timeout before refreshing (to let the user see the success message)
                    setTimeout(() => {
                        // Hide progress overlay
                        viewer.isGenerating = false;
                        showProgress(false);
                        
                        // Refresh history panel to show the new model
                        refreshHistoryPanel();
                    }, 2000);
                    
                    return; // No need to continue polling
                case 'failed':
                    statusTitle.textContent = 'Model generation failed';
                    statusMessage.textContent = data.message || 'An error occurred during model generation. Please try again.';
                    
                    // Hide progress overlay after a delay
                    setTimeout(() => {
                        viewer.isGenerating = false;
                        showProgress(false);
                    }, 3000);
                    
                    return; // No need to continue polling
                default:
                    statusTitle.textContent = 'Model generation status';
                    statusMessage.textContent = `Status: ${data.status}`;
            }
            
            // Continue polling if not completed or failed
            setTimeout(() => pollModelStatus(modelId), 3000);
        })
        .catch(error => {
            console.error('Error polling model status:', error);
            
            // We'll assume the model is still being generated and try again
            // This makes the polling more robust against temporary network issues
            setTimeout(() => pollModelStatus(modelId), 5000); // Longer delay for retry
        });
}

/**
 * Refresh the history panel
 */
function refreshHistoryPanel() {
    console.log("Refreshing history panel...");
    
    // Hide generation status
    const generationStatus = document.getElementById('generation-status');
    if (generationStatus) {
        generationStatus.classList.add('d-none');
    }
    
    // Fetch latest history instead of reloading the page
    _fetch('/api/text-to-3d/history/')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Update the history container with new items
            const historyContainer = document.getElementById('history-container');
            if (!historyContainer) return;
            
            if (data.models && data.models.length > 0) {
                // Clear existing items
                historyContainer.innerHTML = '';
                
                // Add new items
                data.models.forEach(model => {
                    const itemElement = createHistoryItem(model);
                    historyContainer.appendChild(itemElement);
                });
                
                // Re-attach event listeners
                setupHistoryItemListeners();
            } else {
                historyContainer.innerHTML = `
                    <div class="text-center p-4">
                        <p class="text-muted">No models generated yet</p>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error refreshing history panel:', error);
        });
}

/**
 * Create a history item element
 */
function createHistoryItem(model) {
    const div = document.createElement('div');
    div.className = 'history-item';
    div.dataset.id = model.id;
    
    let previewHtml = '';
    if (model.texture_file) {
        previewHtml = `<img src="${model.texture_file}" alt="${model.prompt}" style="max-height: 100%; width: auto;">`;
    } else {
        previewHtml = `<i class="fas fa-cube fa-2x text-secondary"></i>`;
    }
    
    div.innerHTML = `
        <div class="history-item-preview">
            ${previewHtml}
        </div>
        <div class="history-item-info">
            <div class="history-item-title" title="${model.prompt}">${model.prompt.length > 30 ? model.prompt.substring(0, 30) + '...' : model.prompt}</div>
            <div class="history-item-date">${new Date(model.created_at).toLocaleDateString()}</div>
        </div>
    `;
    
    return div;
}

/**
 * Animation loop
 */
function animate() {
    // Prevent multiple animation loops
    if (viewer.isAnimating) {
        console.log("Animation already running, not starting a new loop");
        return;
    }
    
    viewer.isAnimating = true;
    console.log("Animation loop started");
    
    function animationLoop() {
        // Keep animation loop running
        viewer.animationId = requestAnimationFrame(animationLoop);
        
        // Rotate the default cube if visible
        if (viewer.defaultCube && viewer.defaultCube.visible) {
            viewer.defaultCube.rotation.x += 0.01;
            viewer.defaultCube.rotation.y += 0.01;
        }
        
        // Update controls
        if (viewer.controls) {
            viewer.controls.update();
        }
        
        // Render scene
        if (viewer.renderer && viewer.scene && viewer.camera) {
            viewer.renderer.render(viewer.scene, viewer.camera);
        }
    }
    
    // Start the animation loop
    animationLoop();
}

// End of file module initialization
window.addEventListener('load', function() {
    console.log("Window loaded");
    
    // Initialize if not already done
    if (!initialized) {
        initialized = true;
        console.log("Initializing from window load event");
        
        // Initialize Three.js viewport
        initializeViewport();
        
        // Only add default cube if viewport initialized successfully
        if (viewer.scene) {
            // Add default spinning cube to empty scene
            addDefaultCube();
        }
        
        // Initialize UI components and event handlers
        setupEventListeners();
        initializeHistoryPanel();
        loadTrainedModels();
    }
});
