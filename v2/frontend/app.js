/**
 * app.js — Adversarial ML Demo Frontend Logic
 * ============================================
 * Manages the pipeline state, API communication, and UI updates
 * for the adversarial attack/defense demonstration.
 */

// ──────────────── Configuration ────────────────
const API_BASE = "http://localhost:8000";

// ──────────────── Application State ────────────────
const state = {
    originalImage: null,      // base64
    originalResult: null,     // classification result
    adversarialImage: null,   // base64
    adversarialResult: null,  // classification result
    heatmapImage: null,       // base64
    cleanedImage: null,       // base64
    cleanedResult: null,      // classification result
    currentStep: 0,           // 0=upload, 1=classified, 2=attacked, 3=defended
};

// ──────────────── DOM Elements ────────────────
const dom = {
    // Upload
    uploadZone: document.getElementById("uploadZone"),
    fileInput: document.getElementById("fileInput"),

    // Pipeline indicators
    steps: document.querySelectorAll(".pipeline__step"),

    // Sections
    uploadSection: document.getElementById("uploadSection"),
    classificationSection: document.getElementById("classificationSection"),
    attackSection: document.getElementById("attackSection"),
    defenseSection: document.getElementById("defenseSection"),

    // Images
    originalImg: document.getElementById("originalImg"),
    adversarialImg: document.getElementById("adversarialImg"),
    heatmapImg: document.getElementById("heatmapImg"),
    cleanedImg: document.getElementById("cleanedImg"),

    // Prediction containers
    originalPredictions: document.getElementById("originalPredictions"),
    adversarialPredictions: document.getElementById("adversarialPredictions"),
    cleanedPredictions: document.getElementById("cleanedPredictions"),

    // Controls
    epsilonSlider: document.getElementById("epsilonSlider"),
    epsilonValue: document.getElementById("epsilonValue"),
    alphaSlider: document.getElementById("alphaSlider"),
    alphaValue: document.getElementById("alphaValue"),
    itersSlider: document.getElementById("itersSlider"),
    itersValue: document.getElementById("itersValue"),
    defenseMethod: document.getElementById("defenseMethod"),

    // Buttons
    attackBtn: document.getElementById("attackBtn"),
    defendBtn: document.getElementById("defendBtn"),
    resetBtn: document.getElementById("resetBtn"),

    // Status
    statusMessage: document.getElementById("statusMessage"),
};

// ──────────────── Initialization ────────────────
function init() {
    // File upload handlers
    dom.uploadZone.addEventListener("dragover", handleDragOver);
    dom.uploadZone.addEventListener("dragleave", handleDragLeave);
    dom.uploadZone.addEventListener("drop", handleDrop);
    dom.fileInput.addEventListener("change", handleFileSelect);

    // Slider handlers
    dom.epsilonSlider.addEventListener("input", () => {
        dom.epsilonValue.textContent = dom.epsilonSlider.value;
    });
    dom.alphaSlider.addEventListener("input", () => {
        dom.alphaValue.textContent = dom.alphaSlider.value;
    });
    dom.itersSlider.addEventListener("input", () => {
        dom.itersValue.textContent = dom.itersSlider.value;
    });

    // Button handlers
    dom.attackBtn.addEventListener("click", handleAttack);
    dom.defendBtn.addEventListener("click", handleDefend);
    dom.resetBtn.addEventListener("click", handleReset);

    // Load defense methods from backend
    loadDefenseMethods();

    updatePipelineUI();
}

// ──────────────── Load Available Defense Methods ────────────────
async function loadDefenseMethods() {
    // Check if element exists
    if (!dom.defenseMethod) {
        console.error("defenseMethod element not found in DOM");
        return;
    }

    try {
        console.log("Loading defense methods from:", API_BASE + "/defense-methods");
        const response = await fetch(`${API_BASE}/defense-methods`);

        if (!response.ok) {
            console.error(`Failed to fetch defense methods. Status: ${response.status}`);
            console.error("Response text:", await response.text());
            return;
        }

        const data = await response.json();
        console.log("Received data:", data);

        const methods = data.methods || {};
        const descriptions = data.descriptions || {};

        // Clear existing options
        dom.defenseMethod.innerHTML = "";

        // Add options from backend
        Object.entries(methods).forEach(([methodKey, methodName]) => {
            const option = document.createElement("option");
            option.value = methodKey;

            // Get emoji/icon and description from backend
            const description = descriptions[methodKey] || methodName;
            const icon = getIconForMethod(methodKey);

            option.textContent = `${icon} ${methodName}`;
            option.title = description; // Show description on hover
            dom.defenseMethod.appendChild(option);
        });

        console.log("✓ Defense methods loaded successfully. Count:", Object.keys(methods).length);
    } catch (error) {
        console.error("Error loading defense methods:", error);
        console.error("Stack:", error.stack);
    }
}

// Helper function to get emoji icons for methods
function getIconForMethod(methodKey) {
    const icons = {
        "diffusion_restoration": "🎨",  // AI diffusion
        "combined": "⭐",                 // Strongest
        "median_filter": "🟢",            // Green for filter
        "gaussian_blur": "🔵",            // Blue for blur
        "jpeg_compression": "🟠",         // Orange for compression
        "bit_depth_reduction": "🟣",      // Purple for quantization
    };
    return icons[methodKey] || "🛡️";  // Default shield icon
}

// ──────────────── File Upload Handling ────────────────
function handleDragOver(e) {
    e.preventDefault();
    dom.uploadZone.classList.add("upload-zone--active");
}

function handleDragLeave(e) {
    e.preventDefault();
    dom.uploadZone.classList.remove("upload-zone--active");
}

function handleDrop(e) {
    e.preventDefault();
    dom.uploadZone.classList.remove("upload-zone--active");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

async function uploadFile(file) {
    // Validate file type
    if (!file.type.startsWith("image/")) {
        showStatus("Please upload an image file (JPEG, PNG, etc.)", "error");
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showStatus("Image too large. Maximum size is 10MB.", "error");
        return;
    }

    showStatus("⏳ Uploading and classifying image...", "info");
    setButtonLoading(dom.attackBtn, false);

    try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Upload failed");
        }

        const data = await response.json();

        // Update state
        state.originalImage = data.image_base64;
        state.originalResult = data.classification;
        state.currentStep = 1;

        // Update UI
        dom.originalImg.src = `data:image/png;base64,${data.image_base64}`;
        renderPredictions(dom.originalPredictions, data.classification.top5, "");

        showStatus(`✅ Classified as "${data.classification.label}" with ${(data.classification.confidence * 100).toFixed(1)}% confidence`, "success");
        updatePipelineUI();

    } catch (error) {
        showStatus(`❌ ${error.message}`, "error");
    }
}

// ──────────────── Attack Handling ────────────────
async function handleAttack() {
    if (!state.originalImage) {
        showStatus("Please upload an image first.", "warning");
        return;
    }

    const epsilon = parseFloat(dom.epsilonSlider.value);
    const alpha = parseFloat(dom.alphaSlider.value);
    const iterations = parseInt(dom.itersSlider.value);

    showStatus(`⚔️ Generating PGD attack (ε=${epsilon}, α=${alpha}, iters=${iterations})...`, "info");
    setButtonLoading(dom.attackBtn, true);

    try {
        const response = await fetch(`${API_BASE}/attack`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image_base64: state.originalImage,
                epsilon: epsilon,
                alpha: alpha,
                iterations: iterations,
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Attack failed");
        }

        const data = await response.json();

        // Update state
        state.adversarialImage = data.adversarial_image_base64;
        state.adversarialResult = data.adversarial_classification;
        state.heatmapImage = data.heatmap_base64;
        state.currentStep = 2;

        // Update UI
        dom.adversarialImg.src = `data:image/png;base64,${data.adversarial_image_base64}`;
        dom.heatmapImg.src = `data:image/png;base64,${data.heatmap_base64}`;
        renderPredictions(dom.adversarialPredictions, data.adversarial_classification.top5, "--attack");

        const origLabel = state.originalResult.label;
        const advLabel = data.adversarial_classification.label;
        const success = origLabel !== advLabel;

        if (success) {
            showStatus(
                `💥 Attack successful! Model now predicts "${advLabel}" instead of "${origLabel}"`,
                "warning"
            );
        } else {
            showStatus(
                `⚠️ Attack did not change prediction. Try increasing epsilon.`,
                "info"
            );
        }

        updatePipelineUI();

    } catch (error) {
        showStatus(`❌ ${error.message}`, "error");
    } finally {
        setButtonLoading(dom.attackBtn, false);
    }
}

// ──────────────── Defense Handling ────────────────
async function handleDefend() {
    if (!state.adversarialImage) {
        showStatus("Generate an adversarial attack first.", "warning");
        return;
    }

    const method = dom.defenseMethod.value;

    showStatus(`🛡️ Applying ${method.replace(/_/g, " ")} defense...`, "info");
    setButtonLoading(dom.defendBtn, true);

    try {
        const response = await fetch(`${API_BASE}/defend`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image_base64: state.adversarialImage,
                method: method,
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Defense failed");
        }

        const data = await response.json();

        // Update state
        state.cleanedImage = data.cleaned_image_base64;
        state.cleanedResult = data.classification;
        state.currentStep = 3;

        // Update UI
        dom.cleanedImg.src = `data:image/png;base64,${data.cleaned_image_base64}`;
        renderPredictions(dom.cleanedPredictions, data.classification.top5, "--defense");

        const origLabel = state.originalResult.label;
        const cleanLabel = data.classification.label;

        // Extract the top 5 labels from the cleaned predictions
        const top5Labels = data.classification.top5.map(pred => pred.label);

        const perfectRecovery = origLabel === cleanLabel;
        const semanticRecovery = top5Labels.includes(origLabel);

        if (perfectRecovery) {
            showStatus(
                `✅ Defense stripped the attack! Exact prediction recovered: "${cleanLabel}"!`,
                "success"
            );
        } else if (semanticRecovery) {
            showStatus(
                `✅ Defense broke the attack! The model predicts "${cleanLabel}" but your original "${origLabel}" is highly ranked in the Top 5. (Defenses degrade image quality slightly!)`,
                "success"
            );
        } else {
            showStatus(
                `⚠️ Defense altered the prediction to "${cleanLabel}" (original was "${origLabel}"). The attack was broken, but image degradation shifted the class.`,
                "info"
            );
        }

        updatePipelineUI();

    } catch (error) {
        showStatus(`❌ ${error.message}`, "error");
    } finally {
        setButtonLoading(dom.defendBtn, false);
    }
}

// ──────────────── Reset ────────────────
function handleReset() {
    state.originalImage = null;
    state.originalResult = null;
    state.adversarialImage = null;
    state.adversarialResult = null;
    state.heatmapImage = null;
    state.cleanedImage = null;
    state.cleanedResult = null;
    state.currentStep = 0;

    // Clear images
    dom.originalImg.src = "";
    dom.adversarialImg.src = "";
    dom.heatmapImg.src = "";
    dom.cleanedImg.src = "";

    // Clear predictions
    dom.originalPredictions.innerHTML = '<p class="image-display__placeholder">Upload an image to see predictions</p>';
    dom.adversarialPredictions.innerHTML = '<p class="image-display__placeholder">Run attack to see predictions</p>';
    dom.cleanedPredictions.innerHTML = '<p class="image-display__placeholder">Apply defense to see predictions</p>';

    // Reset file input
    dom.fileInput.value = "";

    hideStatus();
    updatePipelineUI();
}

// ──────────────── UI Helpers ────────────────

function renderPredictions(container, top5, variant) {
    container.innerHTML = "";
    top5.forEach((pred, i) => {
        const confidence = (pred.confidence * 100).toFixed(1);
        const div = document.createElement("div");
        div.className = `prediction ${variant ? "prediction" + variant : ""}`;
        div.innerHTML = `
            <span class="prediction__rank">${i + 1}</span>
            <div class="prediction__info">
                <div class="prediction__label">${pred.label}</div>
                <div class="prediction__bar-container">
                    <div class="prediction__bar" style="width: 0%"></div>
                </div>
            </div>
            <span class="prediction__confidence">${confidence}%</span>
        `;
        container.appendChild(div);

        // Animate the bar after a small delay
        setTimeout(() => {
            div.querySelector(".prediction__bar").style.width = `${pred.confidence * 100}%`;
        }, 50 + i * 80);
    });
}

function updatePipelineUI() {
    // Update pipeline step indicators
    dom.steps.forEach((step, index) => {
        const stepNum = index + 1;
        step.classList.remove("pipeline__step--active", "pipeline__step--completed");
        if (stepNum < state.currentStep + 1) {
            step.classList.add("pipeline__step--completed");
        } else if (stepNum === state.currentStep + 1) {
            step.classList.add("pipeline__step--active");
        }
    });

    // Show/hide sections
    dom.classificationSection.classList.toggle("comparison--visible", state.currentStep >= 1);
    dom.attackSection.classList.toggle("comparison--visible", state.currentStep >= 1);
    dom.defenseSection.classList.toggle("comparison--visible", state.currentStep >= 2);

    // Enable/disable buttons
    dom.attackBtn.disabled = state.currentStep < 1;
    dom.defendBtn.disabled = state.currentStep < 2;
    dom.resetBtn.disabled = state.currentStep < 1;
}

function showStatus(message, type = "info") {
    dom.statusMessage.textContent = message;
    dom.statusMessage.className = `status status--visible status--${type}`;
}

function hideStatus() {
    dom.statusMessage.className = "status";
}

function setButtonLoading(btn, loading) {
    if (loading) {
        btn.classList.add("btn--loading");
    } else {
        btn.classList.remove("btn--loading");
    }
}

// ──────────────── Initialize ────────────────
document.addEventListener("DOMContentLoaded", init);
