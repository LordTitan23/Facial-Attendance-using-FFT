import cv2
import os
from tkinter import *
from tkinter import simpledialog, messagebox

# Function to create faces folder if it doesn't exist
def create_faces_folder():
    faces_dir = 'faces'
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
    return faces_dir

# Function to capture and register a new user's face
def register_user(user_name):
    # Open webcam for face capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces_dir = create_faces_folder()

    count = 0  # To keep track of number of images captured

    print(f"Starting face registration for {user_name}. Press 's' to save an image or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangle around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_img = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face_img, (160, 160))  # Resize to 160x160 for Facenet

        # Display the frame with rectangles
        cv2.imshow('Registering - Press "s" to save, "q" to quit', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and len(faces) > 0:
            # Save the captured face image
            count += 1
            file_path = os.path.join(faces_dir, f'{user_name}_{count}.jpg')
            cv2.imwrite(file_path, face_resized)
            print(f"Image saved for {user_name} at {file_path}")

            # Inform the user that the image has been saved
            messagebox.showinfo("Success", f"Image {count} saved for {user_name}.")

            # Optionally, stop after capturing a certain number of images
            if count >= 5:
                print(f"Captured {count} images for {user_name}. Registration complete.")
                messagebox.showinfo("Registration Complete", f"Successfully registered {user_name}.")
                break

        elif key == ord('q'):
            print("Registration cancelled by user.")
            messagebox.showwarning("Cancelled", "Face registration cancelled.")
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Function to capture user name and initiate registration
def capture_user_name():
    # Initialize Tkinter root
    root = Tk()
    root.withdraw()  # Hide the main window

    # Ask the user for their name using a dialog box
    user_name = simpledialog.askstring("Input", "Enter your name:", parent=root)

    if user_name:
        # Start the face registration process
        register_user(user_name)
    else:
        messagebox.showwarning("Input Error", "Name cannot be empty.")

    root.destroy()

if __name__ == "__main__":
    capture_user_name()
