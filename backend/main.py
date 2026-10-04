import sys
import os

# Add parent directory to path so we can import model.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io

from model import load_model, predict_image, CLASSES

app = FastAPI(
    title="Brain Tumor Classifier API",
    description="Classifies brain MRI scans using DenseNet121 + MONAI",
    version="1.0.0"
)

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Load model once when server starts (not on every request)
print("\n" + "=" * 50)
print("BRAIN TUMOR CLASSIFIER - API SERVER")
print("=" * 50)
print("\nLoading DenseNet121 model...")

model = load_model()

print("Model loaded and ready!")
print(f"Classes: {CLASSES}")
print("\nServer starting...")
print("=" * 50 + "\n")


@app.get("/")
def home():
    """Serve the frontend HTML page"""
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Brain Tumor Classifier API is running!", "docs": "/docs"}


@app.get("/health")
def health():
    """Check if the API and model are running"""
    return {
        "status": "healthy",
        "model": "DenseNet121",
        "framework": "MONAI + PyTorch",
        "classes": CLASSES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a brain MRI image and get a tumor classification prediction.

    Returns:
        prediction: the predicted class name
        confidence: model confidence as a percentage
        probabilities: probability % for each of the 4 classes
    """

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file (JPG, PNG)"
        )

    try:
        # Read and open the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run prediction using model.py
        prediction, confidence, probabilities = predict_image(image, model)

        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {
                k: round(v, 2) for k, v in probabilities.items()
            },
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
