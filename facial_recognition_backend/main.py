from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tensorflow as tf
from deepface import DeepFace
from datetime import datetime, timedelta
import json
import os
import base64
from sklearn.decomposition import PCA

# GPU configuration for TensorFlow (memory growth)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Limit GPU memory growth to avoid memory exhaustion
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error configuring GPU memory: {str(e)}")

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Global Variables
DATA_FILE = "data.json"  # JSON file for storing user data and attendance logs
ATTENDANCE_TIMEOUT = 10  # Minutes before re-marking allowed
CONFIDENCE_THRESHOLD = 70  # Minimum confidence percentage for recognition
STANDARDIZED_SIZE = (160, 160)  # Size for standardizing face images
CPT_COMPONENTS = 50  # Number of components for CPT (PCA)

# Initialize data file
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump({"users": {}, "attendance_logs": []}, f)

# Load data from the file
def load_data():
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"users": {}, "attendance_logs": []}

# Save data to the file
def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Helper: Ensure image is in 3-channel BGR format and uint8 depth
def ensure_bgr_format(image):
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    
    if len(image.shape) == 2:  # Grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # BGRA
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image

# Decode base64 image
def decode_image(image_data):
    try:
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        
        image_bytes = base64.b64decode(image_data)
        np_image = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        raise ValueError(f"Invalid image format: {str(e)}")

# Apply FFT preprocessing to face image
def preprocess_with_fft_and_cpt(face_image):
    try:
        if face_image is None or face_image.size == 0:
            raise ValueError("No valid face detected.")
        
        resized_face = cv2.resize(face_image, STANDARDIZED_SIZE)
        gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
        
        # Apply FFT to the grayscale face
        f = np.fft.fft2(gray_face)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        
        # Normalize the spectrum
        normalized_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply PCA (CPT) to reduce dimensionality
        # Flatten the 2D image into a 1D vector
        flattened_spectrum = normalized_spectrum.flatten().reshape(-1, 1)
        
        pca = PCA(n_components=CPT_COMPONENTS)
        transformed_spectrum = pca.fit_transform(flattened_spectrum.T).T
        
        # Reshape back to the original shape
        transformed_spectrum_reshaped = transformed_spectrum.reshape(normalized_spectrum.shape)
        
        # Merge to create a 3-channel image for consistency
        pseudo_rgb = cv2.merge([transformed_spectrum_reshaped] * 3)  # Create 3-channel image
        
        return pseudo_rgb.astype(np.uint8)
    except Exception as e:
        raise ValueError(f"Error in preprocessing face with FFT and CPT: {str(e)}")

# Register a new user
@app.post("/register/") 
async def register_user(data: dict = Body(...)):
    username = data.get("username")
    image = data.get("image")

    if not username or not image:
        return JSONResponse({"message": "Username and image are required."}, status_code=400)

    data_store = load_data()

    if username in data_store["users"]:
        return JSONResponse({"message": f"User '{username}' is already registered."}, status_code=400)

    try:
        frame = decode_image(image)
        frame = ensure_bgr_format(frame)

        # Extract faces
        faces = DeepFace.extract_faces(img_path=frame, detector_backend="opencv", enforce_detection=True)
        if not faces:
            return JSONResponse({"message": "No face detected in the image."}, status_code=400)

        # Process first detected face
        face_detected = faces[0]["face"]
        face_detected = ensure_bgr_format(face_detected)
        fft_cpt_preprocessed = preprocess_with_fft_and_cpt(face_detected)

        data_store["users"][username] = [fft_cpt_preprocessed.tolist()]
        save_data(data_store)

        return JSONResponse({"message": f"User '{username}' registered successfully."})
    except Exception as e:
        return JSONResponse({"message": f"Error: {str(e)}"}, status_code=500)

# Recognize and mark attendance
@app.post("/recognize/") 
async def recognize_user(data: dict = Body(...)):
    image = data.get("image")

    if not image:
        return JSONResponse({"message": "Image is required."}, status_code=400)

    data_store = load_data()

    try:
        frame = decode_image(image)
        frame = ensure_bgr_format(frame)
        fft_cpt_frame = preprocess_with_fft_and_cpt(frame)

        response = {"message": "No matching face found.", "accuracy": None}

        for username, user_faces in data_store["users"].items():
            for stored_face in user_faces:
                try:
                    result = DeepFace.verify(np.array(stored_face), fft_cpt_frame, enforce_detection=False)
                    if result["verified"]:
                        accuracy = 100 - result["distance"] * 100
                        if accuracy >= CONFIDENCE_THRESHOLD:
                            now = datetime.now()
                            recent_logs = [
                                log for log in data_store["attendance_logs"]
                                if log["user"] == username and 
                                datetime.strptime(log["time"], "%Y-%m-%d %H:%M:%S") > now - timedelta(minutes=ATTENDANCE_TIMEOUT)
                            ]

                            if recent_logs:
                                response["message"] = f"Attendance already marked for '{username}'."
                                return JSONResponse(response)

                            data_store["attendance_logs"].append({
                                "user": username,
                                "time": now.strftime("%Y-%m-%d %H:%M:%S")
                            })
                            save_data(data_store)

                            response["message"] = f"Attendance marked for '{username}' successfully."
                            response["accuracy"] = accuracy
                            return JSONResponse(response)
                except Exception as e:
                    continue

        return JSONResponse(response)
    except Exception as e:
        return JSONResponse({"message": f"Error: {str(e)}"}, status_code=500)

# View Attendance Records
@app.get("/attendance_records/") 
async def view_attendance_records():
    data = load_data()
    return JSONResponse({"records": data["attendance_logs"]})

# Additional health check endpoint
@app.get("/health/") 
async def health_check():
    return JSONResponse({"status": "healthy"})
