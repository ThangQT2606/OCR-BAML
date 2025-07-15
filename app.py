import base64
import io
import uuid
import cv2
import asyncio
import time
from typing import Optional
import numpy as np
import fastapi
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder

# Import functions from async_engine
from async_engine import init_cr, llm_predict

app = FastAPI(
    title="Invoice LLM API",
    description="API for extracting information from invoice images using LLM",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize client registry with default LLM
default_cr = init_cr(LLM_CR="Gemini_2_0_pro")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: fastapi.Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({
            "returncode": {
                "code": 422,
                "message": "Request json has syntax error"
            },
            "output": {"result": [], "timing": 0.0}
        }),
    )

@app.get("/")
async def root():
    return {"message": "Invoice LLM API is running"}


@app.post("/extract_invoice")
async def extract_invoice(file: UploadFile = File(...), llm_model: Optional[str] = "Gemini_2_0_pro"):
    """
    Extract information from an invoice image
    
    - **file**: The invoice image file
    - **llm_model**: Optional LLM model name (default: Gemini_2_0_pro)
    
    Returns extracted information from the invoice
    """
    start_time = time.time()
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not an image")
    
    try:
        # Read image from the uploaded file
        contents = await file.read()
        image_stream = io.BytesIO(contents)
        image_array = np.frombuffer(image_stream.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
            
        # Get client registry for specified LLM model
        cr = init_cr(LLM_CR=llm_model)
        
        # Generate a unique ID for this request
        request_id = str(uuid.uuid4())
        
        # Process the image
        results = await llm_predict(
            uuid=request_id,
            files_name=file.filename,
            images=[image],
            cr=cr
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add processing time to the response
        response = {
            "results": results,
            "processing_time_seconds": processing_time,
            "request_id": request_id
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/models")
async def list_models():
    """List available LLM models"""
    # These are example models, update based on what your system actually supports
    models = ["Gemini_2_0_pro", "Gemini_1_5_pro", "GPT_4o"]
    return {"models": models}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
