import cv2
import os
import numpy as np
from datetime import datetime, timedelta
from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox
from deepface import DeepFace
from mtcnn import MTCNN
import threading
import time

# Suppress TensorFlow logging for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize MTCNN for face detection and load the DeepFace model once
detector = MTCNN()
deepface_model = DeepFace.build_model("Facenet")  # Load the model once

# CNN-based face recognition function with optimized processing
def recognize_face_cnn(face_img):
    try:
        # Convert face image to grayscale and apply FFT preprocessing
        gray_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(gray_img)
        f_transform = np.fft.fftshift(f_transform)
        
        # Apply a low pass filter for smoothing
        rows, cols = gray_img.shape
        crow, ccol = rows // 2, cols // 2
        radius = 30
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
        f_transform *= mask
        
        # Inverse FFT to reconstruct filtered image
        f_ishift = np.fft.ifftshift(f_transform)
        filtered_img = np.abs(np.fft.ifft2(f_ishift)).astype(np.uint8)
        
        # Save processed face image temporarily to disk for DeepFace
        temp_face_path = 'temp_filtered_face.jpg'
        cv2.imwrite(temp_face_path, filtered_img)

        # Get face embeddings from the processed image path
        embedding_obj = DeepFace.represent(
            img_path=temp_face_path,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend='mtcnn'
        )

        # Remove the temporary image file
        os.remove(temp_face_path)

        # Extract embedding if available
        if embedding_obj and isinstance(embedding_obj, list):
            input_embedding = embedding_obj[0]['embedding']
            return np.array(input_embedding)
        
        return None
    except Exception as e:
        print(f"Error in recognize_face_cnn: {e}")
        return None

# Calculate cosine similarity between embeddings
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Precompute embeddings for registered faces using DeepFace
def load_registered_faces():
    registered_embeddings = {}
    faces_folder = 'faces'
    if not os.path.exists(faces_folder):
        return registered_embeddings

    for filename in os.listdir(faces_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(faces_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Apply CNN to get face embeddings
            try:
                embedding = recognize_face_cnn(img)
                if embedding is not None:
                    registered_name = os.path.splitext(filename)[0].rsplit('_', 1)[0]
                    registered_embeddings[registered_name] = embedding
            except Exception as e:
                print(f"Error in load_registered_faces: {e}")

    return registered_embeddings

# Mark attendance in the log file
def mark_attendance(user_name):
    data_folder = 'data'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    attendance_file = os.path.join(data_folder, 'attendance_log.csv')
    with open(attendance_file, 'a') as f:
        now = datetime.now()
        dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'{user_name},{dt_string}\n')

# Compare face embeddings to identify face
def recognize_face(input_embedding, registered_embeddings, threshold=0.8):
    if input_embedding is None:
        return None, None

    best_match = None
    highest_similarity = -1

    for name, reg_embedding in registered_embeddings.items():
        similarity = cosine_similarity(input_embedding, reg_embedding)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = name

    if highest_similarity >= threshold:
        accuracy = highest_similarity
        return best_match, accuracy

    return None, None

# Main GUI and video processing class
class FacialAttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Attendance System")
        self.root.geometry("1024x768")

        # High-resolution GUI and improved fonts
        self.label = Label(root, text="Facial Attendance System", font=("Helvetica", 32, "bold"), fg="blue")
        self.label.pack(pady=20)

        self.video_frame = Label(root)
        self.video_frame.pack()

        self.accuracy_label = Label(root, text="", font=("Helvetica", 20, "bold"), fg="green")
        self.accuracy_label.pack(pady=10)

        # Message display label for attendance marking
        self.message_label = Label(root, text="", font=("Helvetica", 20, "bold"), fg="orange")
        self.message_label.pack(pady=10)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.root.destroy()
            return

        # Load registered faces and their embeddings
        self.registered_embeddings = load_registered_faces()
        if not self.registered_embeddings:
            messagebox.showwarning("Warning", "No registered faces found.")
            self.root.destroy()
            return

        self.attendance_records = {}
        self.current_users = set()

        # Control flags and buffers
        self.running = True
        self.frame_skip = 3  # Optimized to process every third frame
        self.frame_count = 0

        # Start video processing in a separate thread
        self.thread = threading.Thread(target=self.update_frame, daemon=True)
        self.thread.start()

        # Ensure proper thread termination on closing the window
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_frame(self):
        try:
            while self.running:
                ret, frame = self.cap.read()
                now = datetime.now()

                if not ret:
                    time.sleep(0.03)
                    continue

                self.frame_count += 1
                if self.frame_count % self.frame_skip != 0:
                    time.sleep(0.03)
                    continue

                frame = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces and recognize users
                try:
                    detections = detector.detect_faces(rgb_frame)
                except Exception:
                    detections = []

                if len(detections) == 0:
                    self.current_users.clear()
                    self.accuracy_label.config(text="")
                    self.message_label.config(text="No user detected", fg="red")
                else:
                    for detection in detections:
                        x, y, w, h = detection['box']
                        x, y = max(0, x), max(0, y)
                        face_img = rgb_frame[y:y+h, x:x+w]

                        input_embedding = recognize_face_cnn(face_img)
                        recognized_user, accuracy = recognize_face(input_embedding, self.registered_embeddings, threshold=0.8)

                        if recognized_user:
                            if recognized_user in self.attendance_records:
                                last_attendance_time = self.attendance_records[recognized_user]
                                if (now - last_attendance_time) < timedelta(minutes=10):
                                    self.message_label.config(text=f"Attendance already marked for {recognized_user}", fg="orange")
                                    self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}")
                                    continue

                            mark_attendance(recognized_user)
                            self.attendance_records[recognized_user] = now
                            self.message_label.config(text=f"Attendance marked for {recognized_user} successfully", fg="green")
                            self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}")
                        else:
                            self.message_label.config(text="Unknown user", fg="red")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)

                time.sleep(0.03)
        except Exception as e:
            print(f"Error in update_frame: {e}")
        finally:
            self.cap.release()

    def on_closing(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        self.root.destroy()

# Main execution
if __name__ == "__main__":
    root = Tk()
    app = FacialAttendanceSystem(root)
    root.mainloop()
  