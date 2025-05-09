from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import os
from datetime import datetime
import cv2
import numpy as np

from app.utils.inference import YOLOv8FireDetector
from app.utils.image_processing import bytes_to_image, image_to_bytes, draw_detections

app = FastAPI()



# Setup directories
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR/ "app" / "models" / "yolov8_fire.onnx"
TEMP_DIR = BASE_DIR/ "app" / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# Initialize detector
detector = YOLOv8FireDetector(str(MODEL_PATH))

# Mount temp directory as static files
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")

# Mount static files
app.mount("/static", StaticFiles(directory=BASE_DIR/ "app" / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR/ "app" / "templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, detail="Invalid image file")
    
    try:
        # Process image
        image_bytes = await file.read()
        image = bytes_to_image(image_bytes)
        
        # Run detection
        image, detections = detector.predict(image)
        
        # Draw detections
        result_image = draw_detections(image, detections)
        
        # Save result with proper filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{timestamp}.jpg"
        output_path = TEMP_DIR / filename
        cv2.imwrite(str(output_path), result_image)
        
        # Format response with correct URL
        return {
            "image_url": f"/temp/{filename}",
            "detections": [
                {"label": label, "confidence": float(conf)}
                for label, conf, _ in detections
            ]
        }
    
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/stream")
async def stream_video(frame: UploadFile = File(...)):
    try:
        image_bytes = await frame.read()
        image = bytes_to_image(image_bytes)
        
        # Run detection (using 'predict' instead of 'detect')
        image, detections = detector.predict(image)
        
        if detections:
            # Draw detections
            result_image = draw_detections(image, detections)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = TEMP_DIR / f"stream_result_{timestamp}.jpg"
            cv2.imwrite(str(output_path), result_image)
            
            return {
                "detections": [
                    {"label": label, "confidence": float(conf)}
                    for label, conf, _ in detections
                ],
                "image_url": f"/temp/stream_result_{timestamp}.jpg"
            }
        
        return {"detections": []}
    
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.on_event("shutdown")
def cleanup():
    # Clean temp directory on shutdown
    for file in TEMP_DIR.glob("*.jpg"):
        try:
            file.unlink()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))