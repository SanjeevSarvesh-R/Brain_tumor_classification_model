const API_URL = "http://localhost:8000";

const CLASS_ICONS = {
    glioma:     "🔴",
    meningioma: "🟠",
    pituitary:  "🟡",
    notumor:    "🟢"
};

const CLASS_COLORS = {
    glioma:     "#fc8181",
    meningioma: "#f6ad55",
    pituitary:  "#f6e05e",
    notumor:    "#68d391"
};

// DOM elements
const uploadZone   = document.getElementById("uploadZone");
const fileInput    = document.getElementById("fileInput");
const uploadIconWrap = document.getElementById("uploadIconWrap");
const previewWrap  = document.getElementById("previewWrap");
const previewImg   = document.getElementById("previewImg");
const previewFilename = document.getElementById("previewFilename");
const changeBtn    = document.getElementById("changeBtn");
const predictBtn   = document.getElementById("predictBtn");
const btnLoader    = document.getElementById("btnLoader");
const resultCard   = document.getElementById("resultCard");
const statusBadge  = document.getElementById("statusBadge");
const statusText   = document.getElementById("statusText");

// ─── Check server health on load ──────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(3000) });
        if (res.ok) {
            statusBadge.classList.add("online");
            statusText.textContent = "Model Ready";
        } else {
            throw new Error();
        }
    } catch {
        statusText.textContent = "Server Offline";
    }
}
checkHealth();

// ─── Click upload zone to open file picker ────────────────────
uploadZone.addEventListener("click", (e) => {
    if (e.target === changeBtn) return;
    fileInput.click();
});

changeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
});

// ─── Drag and drop ────────────────────────────────────────────
uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
        handleFileSelected(file);
    }
});

// ─── File input change ────────────────────────────────────────
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) handleFileSelected(file);
});

function handleFileSelected(file) {
    // Show image preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewFilename.textContent = file.name;
        uploadIconWrap.style.display = "none";
        previewWrap.style.display = "block";
    };
    reader.readAsDataURL(file);

    // Enable predict button
    predictBtn.disabled = false;

    // Hide old result
    resultCard.style.display = "none";
}

// ─── Predict ──────────────────────────────────────────────────
predictBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    // Show loading state
    predictBtn.querySelector(".btn-text").textContent = "Analyzing...";
    predictBtn.querySelector(".btn-icon").style.display = "none";
    btnLoader.style.display = "block";
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
            const err = await response.json();
            throw new Error(err.detail || "Prediction failed");
        }

        const data = await response.json();
        showResult(data);

    } catch (error) {
        alert(`Error: ${error.message}\n\nMake sure the backend server is running:\ncd backend && uvicorn main:app --reload`);
    } finally {
        // Reset button
        predictBtn.querySelector(".btn-text").textContent = "Analyze MRI Scan";
        predictBtn.querySelector(".btn-icon").style.display = "inline";
        btnLoader.style.display = "none";
        predictBtn.disabled = false;
    }
});

// ─── Show result ──────────────────────────────────────────────
function showResult(data) {
    const { prediction, confidence, probabilities } = data;

    // Timestamp
    document.getElementById("resultTimestamp").textContent =
        new Date().toLocaleTimeString();

    // Main prediction
    document.getElementById("predictionIcon").textContent =
        CLASS_ICONS[prediction] || "🔵";

    const predEl = document.getElementById("predictionValue");
    predEl.textContent = prediction === "notumor" ? "No Tumor Detected" : prediction;
    predEl.style.color = CLASS_COLORS[prediction] || "white";

    document.getElementById("confidenceValue").textContent = `${confidence}%`;

    // Confidence bar
    const fill = document.getElementById("confidenceFill");
    fill.style.width = "0%";
    setTimeout(() => { fill.style.width = `${confidence}%`; }, 100);

    // Per-class probabilities
    const probList = document.getElementById("probList");
    probList.innerHTML = "";

    // Sort by probability descending
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    sorted.forEach(([cls, prob]) => {
        const isTop = cls === prediction;
        const item = document.createElement("div");
        item.className = "prob-item";
        item.innerHTML = `
            <span class="prob-name">${CLASS_ICONS[cls] || "•"} ${cls === "notumor" ? "No Tumor" : cls}</span>
            <div class="prob-bar-track">
                <div class="prob-bar-fill ${isTop ? "top" : "other"}" 
                     style="width: 0%"
                     data-width="${prob}"></div>
            </div>
            <span class="prob-pct ${isTop ? "top" : ""}">${prob}%</span>
        `;
        probList.appendChild(item);
    });

    // Show result card
    resultCard.style.display = "block";
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });

    // Animate bars after render
    setTimeout(() => {
        document.querySelectorAll(".prob-bar-fill").forEach(bar => {
            bar.style.width = bar.dataset.width + "%";
        });
    }, 200);
}
