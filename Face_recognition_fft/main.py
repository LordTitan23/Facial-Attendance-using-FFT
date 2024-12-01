import os
import subprocess
from tkinter import *

def register_user():
    subprocess.run(["python", "scripts/register.py"])

def mark_attendance():
    subprocess.run(["python", "scripts/recog.py"])

def show_attendance_log():
    subprocess.run(["python", "scripts/attendance_log.py"])

# Create the main window
root = Tk()
root.title("Facial Attendance System")
root.geometry("400x300")

# Create the buttons
Label(root, text="Facial Attendance System", font=("Arial", 20)).pack(pady=20)

Button(root, text="Register New User", command=register_user, width=20, height=2).pack(pady=10)
Button(root, text="Mark Attendance", command=mark_attendance, width=20, height=2).pack(pady=10)
Button(root, text="Attendance Log", command=show_attendance_log, width=20, height=2).pack(pady=10)

# Run the GUI loop
root.mainloop()
