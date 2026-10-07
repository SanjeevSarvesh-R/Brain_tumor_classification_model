/**
 * BRAINVERSE — Frontend Application Logic
 * Clinical healthcare interface for brain MRI tumor classification.
 */

// Dynamic API URL: Works when served locally, on custom ports, or across local networks
const API_URL = (window.location.origin && window.location.origin !== "null") 
    ? window.location.origin 
    : "http://localhost:8000";

// Diagnostic Class Semantic Configuration (Muted Healthcare Tones)
const DIAGNOSTIC_CLASSES = {
    glioma: {
        label: "Glioma",
        color: "#B91C1C",       // Muted clinical red
        bgColor: "#FEF2F2",
        borderColor: "#FECACA",
        badge: "🔴"
    },
    meningioma: {
        label: "Meningioma",
        color: "#B45309",       // Muted clinical amber
        bgColor: "#FFFBEB",
        borderColor: "#FDE68A",
        badge: "🟠"
    },
    pituitary: {
        label: "Pituitary Tumor",
        color: "#1D4ED8",       // Muted clinical blue
        bgColor: "#EFF6FF",
        borderColor: "#BFDBFE",
        badge: "🔵"
    },
    notumor: {
        label: "No Tumor Detected",
        color: "#15803D",       // Muted clinical green
        bgColor: "#F0FDF4",
        borderColor: "#BBF7D0",
        badge: "🟢"
    }
};

// DOM Element References
const uploadZone       = document.getElementById("uploadZone");
const fileInput        = document.getElementById("fileInput");
const browseBtn        = document.getElementById("browseBtn");
const uploadIconWrap   = document.getElementById("uploadIconWrap");
const previewWrap      = document.getElementById("previewWrap");
const previewImg       = document.getElementById("previewImg");
const previewFilename  = document.getElementById("previewFilename");
const changeBtn        = document.getElementById("changeBtn");
const predictBtn       = document.getElementById("predictBtn");
const btnLoader        = document.getElementById("btnLoader");
const resultCard       = document.getElementById("resultCard");
const statusBadge      = document.getElementById("statusBadge");
const statusText       = document.getElementById("statusText");
const mobileToggle     = document.getElementById("mobileToggle");
const mobileDrawer     = document.getElementById("mobileDrawer");

// ─── 1. Mobile Menu Drawer Toggle ─────────────────────────────────────────
if (mobileToggle && mobileDrawer) {
    mobileToggle.addEventListener("click", () => {
        const isExpanded = mobileToggle.getAttribute("aria-expanded") === "true";
        mobileToggle.setAttribute("aria-expanded", !isExpanded);
        mobileDrawer.classList.toggle("active");
    });

    // Close mobile drawer upon navigating
    mobileDrawer.querySelectorAll(".mobile-nav-item, .mobile-link, .mobile-cta").forEach(item => {
        item.addEventListener("click", () => {
            mobileDrawer.classList.remove("active");
            mobileToggle.setAttribute("aria-expanded", "false");
        });
    });
}

// ─── 2. 3-Second Header/Hero Image Carousel ───────────────────────────────
const TOTAL_SLIDES = 3;
let currentSlide = 0;
const slideTelemetryTitles = [
    "PATIENT CONSULTATION • INTAKE",
    "NEURO-RADIOLOGY & MRI DIAGNOSTICS",
    "CLINICAL DEEP LEARNING EVALUATION"
];

function setHeroSlide(index) {
    currentSlide = (index + TOTAL_SLIDES) % TOTAL_SLIDES;

    // 1. Update background slideshow
    const bgSlides = document.querySelectorAll(".bg-slide");
    bgSlides.forEach((slide, idx) => {
        slide.classList.toggle("active", idx === currentSlide);
    });

    // 2. Update indicator dots
    const dots = document.querySelectorAll(".carousel-dot");
    dots.forEach((dot, idx) => {
        dot.classList.toggle("active", idx === currentSlide);
    });

    // 3. Update telemetry label
    const teleEl = document.getElementById("slideTelemetryTitle");
    if (teleEl) {
        teleEl.textContent = slideTelemetryTitles[currentSlide];
    }
}

// 3-second (3000ms) automatic transition cycle
let carouselTimer = setInterval(() => {
    setHeroSlide(currentSlide + 1);
}, 3000);

// Allow interactive navigation via dots
document.querySelectorAll(".carousel-dot").forEach((dot) => {
    dot.addEventListener("click", (e) => {
        e.stopPropagation();
        clearInterval(carouselTimer);
        const targetIndex = parseInt(dot.dataset.slide, 10);
        setHeroSlide(targetIndex);
        carouselTimer = setInterval(() => {
            setHeroSlide(currentSlide + 1);
        }, 3000);
    });
});

// ─── 3. System Health Status Check ────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(4000) });
        if (res.ok) {
            if (statusBadge) {
                statusBadge.classList.remove("offline");
                statusBadge.classList.add("online");
            }
            if (statusText) statusText.textContent = "AI Model Ready";
        } else {
            throw new Error("Service error");
        }
    } catch {
        if (statusBadge) {
            statusBadge.classList.remove("online");
            statusBadge.classList.add("offline");
        }
        if (statusText) statusText.textContent = "Service Unavailable";
    }
}

// ─── 4. Clinical Portal Sign In Modal Handlers ────────────────────────────
const signInBtn        = document.getElementById("signInBtn");
const mobileSignInLink = document.getElementById("mobileSignInLink");
const signInModal      = document.getElementById("signInModal");
const closeSignInModal = document.getElementById("closeSignInModal");
const signInForm       = document.getElementById("signInForm");

function openSignInModal() {
    if (signInModal) signInModal.style.display = "flex";
}

function closeSignInModalDialog() {
    if (signInModal) signInModal.style.display = "none";
}

if (signInBtn) {
    signInBtn.addEventListener("click", (e) => {
        e.preventDefault();
        openSignInModal();
    });
}

if (mobileSignInLink) {
    mobileSignInLink.addEventListener("click", (e) => {
        e.preventDefault();
        if (mobileDrawer) mobileDrawer.classList.remove("active");
        openSignInModal();
    });
}

if (closeSignInModal) {
    closeSignInModal.addEventListener("click", closeSignInModalDialog);
}

if (signInModal) {
    signInModal.addEventListener("click", (e) => {
        if (e.target === signInModal) {
            closeSignInModalDialog();
        }
    });
}

window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && signInModal && signInModal.style.display === "flex") {
        closeSignInModalDialog();
    }
});

if (signInForm) {
    signInForm.addEventListener("submit", (e) => {
        e.preventDefault();
        alert("Clinical session authenticated successfully. Welcome to BrainVerse.");
        closeSignInModalDialog();
    });
}

checkHealth();

// ─── 3. File Selection & Drag-and-Drop Handlers ───────────────────────────
// Trigger file picker when clicking the dropzone (except when clicking Change Image)
uploadZone.addEventListener("click", (e) => {
    if (e.target === changeBtn) return;
    fileInput.click();
});

// Keyboard accessibility: Enter or Space on dropzone
uploadZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
    }
});

if (browseBtn) {
    browseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });
}

changeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
});

// Drag and drop event listeners
uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) {
        validateAndProcessFile(file);
    }
});

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
        validateAndProcessFile(file);
    }
});

function validateAndProcessFile(file) {
    const validExtensions = [".jpg", ".jpeg", ".png"];
    const fileName = file.name.toLowerCase();
    const isValidExtension = validExtensions.some(ext => fileName.endsWith(ext));
    const isValidMime = file.type.startsWith("image/");

    if (!isValidExtension && !isValidMime) {
        alert("Please select a supported brain MRI image (JPG, JPEG, or PNG).");
        return;
    }

    // Read and display local image preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewFilename.textContent = file.name;
        uploadIconWrap.style.display = "none";
        previewWrap.style.display = "block";
    };
    reader.readAsDataURL(file);

    // Enable predict button and reset any prior results
    predictBtn.disabled = false;
    resultCard.style.display = "none";
}

// ─── 4. Inference Execution (POST /predict) ───────────────────────────────
predictBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    // Set clinical loading state
    const btnTextElem = predictBtn.querySelector(".btn-text");
    btnTextElem.textContent = "Analyzing MRI Scan...";
    btnLoader.style.display = "inline-block";
    predictBtn.disabled = true;
    resultCard.style.display = "none";

    try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            let errorDetail = "Inference failed";
            try {
                const err = await response.json();
                errorDetail = err.detail || errorDetail;
            } catch (_) {}
            throw new Error(errorDetail);
        }

        const data = await response.json();
        showResult(data);

    } catch (error) {
        alert(`Prediction Error: ${error.message}\n\nPlease verify that the backend API service is running on ${API_URL}`);
    } finally {
        // Restore button state
        btnTextElem.textContent = "Analyze MRI Scan →";
        btnLoader.style.display = "none";
        predictBtn.disabled = false;
    }
});

// ─── 5. Clinical Result Rendering ─────────────────────────────────────────
function showResult(data) {
    const { prediction, confidence, probabilities, filename } = data;
    const classMeta = DIAGNOSTIC_CLASSES[prediction] || {
        label: prediction,
        color: "#059669",
        bgColor: "#ECFDF5",
        borderColor: "#A7F3D0",
        badge: "🟢"
    };

    // Format timestamp
    const now = new Date();
    document.getElementById("resultTimestamp").textContent = now.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
    });

    // Primary diagnostic category
    const iconEl = document.getElementById("predictionIcon");
    iconEl.textContent = classMeta.badge;

    const valueEl = document.getElementById("predictionValue");
    valueEl.textContent = classMeta.label;
    valueEl.style.color = classMeta.color;

    // Confidence metrics
    document.getElementById("confidenceValue").textContent = `${confidence.toFixed(2)}%`;

    // Overall confidence progress fill
    const fill = document.getElementById("confidenceFill");
    fill.style.width = "0%";
    setTimeout(() => {
        fill.style.width = `${confidence}%`;
        fill.style.backgroundColor = classMeta.color;
    }, 100);

    // Render per-class probabilities
    const probList = document.getElementById("probList");
    probList.innerHTML = "";

    // Sort descending by probability
    const sortedProbabilities = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    sortedProbabilities.forEach(([clsKey, probValue]) => {
        const isTopPrediction = clsKey === prediction;
        const currentMeta = DIAGNOSTIC_CLASSES[clsKey] || {
            label: clsKey,
            color: "#64748B",
            badge: "•"
        };

        const item = document.createElement("div");
        item.className = "prob-item";
        item.innerHTML = `
            <span class="prob-name" style="${isTopPrediction ? `color: ${currentMeta.color}; font-weight: 700;` : ''}">
                ${currentMeta.badge} ${currentMeta.label}
            </span>
            <div class="prob-bar-track">
                <div class="prob-bar-fill ${isTopPrediction ? 'top' : ''}" 
                     style="width: 0%; ${isTopPrediction ? `background-color: ${currentMeta.color};` : ''}"
                     data-width="${probValue}"></div>
            </div>
            <span class="prob-pct ${isTopPrediction ? 'top' : ''}" style="${isTopPrediction ? `color: ${currentMeta.color};` : ''}">
                ${probValue.toFixed(2)}%
            </span>
        `;
        probList.appendChild(item);
    });

    // Filename reference
    const refFilenameEl = document.getElementById("resultFilename");
    if (refFilenameEl) {
        refFilenameEl.textContent = filename || (fileInput.files[0] ? fileInput.files[0].name : "scan.jpg");
    }

    // Display report card and scroll smoothly into view
    resultCard.style.display = "block";
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });

    // Animate probability distribution bars
    setTimeout(() => {
        document.querySelectorAll(".prob-bar-fill").forEach(bar => {
            bar.style.width = bar.dataset.width + "%";
        });
    }, 200);
}
