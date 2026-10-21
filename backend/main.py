from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from backend.services.model_service import model_service
from backend.services.prediction_service import format_prediction_response

app = FastAPI(
    title="Lung Cancer AI API",
    description="Backend API for Lung Cancer classification models",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Load models when the server starts
    model_service.load_models()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": model_service.is_loaded
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Verify it's actually an image
        image.verify()
        
        # Re-open because verify() closes it
        image = Image.open(BytesIO(contents))
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
    try:
        # Run inference
        model_results = model_service.predict_all(image)
        
        # Format response
        response = format_prediction_response(model_results)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {str(e)}")

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
