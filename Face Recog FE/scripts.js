// DOM Element Selections
const videoFeed = document.getElementById("videoFeed");
const registerBtn = document.getElementById("registerBtn");
const markAttendanceBtn = document.getElementById("markAttendanceBtn");
const usernameInput = document.getElementById("username");
const resultDiv = document.getElementById("result");
const errorMessage = document.getElementById("errorMessage");
const attendanceList = document.getElementById("attendanceList");
const canvas = document.getElementById("canvas");
const ctx = canvas?.getContext("2d");
const confidenceDisplay = document.getElementById("confidenceDisplay");

// Configuration
const API_BASE_URL = 'http://localhost:8000';

// Start the webcam feed
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true,
        });
        videoFeed.srcObject = stream;

        // Adjust canvas dimensions after video feed is ready
        videoFeed.onloadedmetadata = () => {
            canvas.width = videoFeed.videoWidth;
            canvas.height = videoFeed.videoHeight;
        };
    } catch (error) {
        errorMessage.textContent = 'Error accessing webcam: ' + error.message;
    }
}

// Capture frame and send to backend for recognition
async function captureFrame() {
    ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
    const base64Image = canvas.toDataURL("image/jpeg");

    try {
        const response = await fetch(`${API_BASE_URL}/recognize/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: base64Image })
        });

        const data = await response.json();

        if (data.message) {
            resultDiv.textContent = data.message;

            if (data.accuracy !== null) {
                confidenceDisplay.textContent = `Confidence: ${data.accuracy.toFixed(2)}%`;
            }
        }
    } catch (error) {
        errorMessage.textContent = 'Error during recognition: ' + error.message;
    }
}

// Register User
async function registerUser() {
    const username = usernameInput.value.trim();

    if (!username) {
        errorMessage.textContent = "Please enter a username.";
        return;
    }

    const image = canvas.toDataURL("image/jpeg");

    try {
        const response = await fetch(`${API_BASE_URL}/register/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, image })
        });

        const data = await response.json();

        if (data.message) {
            resultDiv.textContent = data.message;
        }
    } catch (error) {
        errorMessage.textContent = 'Error during registration: ' + error.message;
    }
}

// Get Attendance Logs
async function getAttendanceLogs() {
    try {
        const response = await fetch(`${API_BASE_URL}/attendance_records/`);
        const data = await response.json();

        attendanceList.innerHTML = data.records.map(record => 
            `<li>${record.user} at ${record.time}</li>`
        ).join('');
    } catch (error) {
        errorMessage.textContent = 'Error fetching attendance records: ' + error.message;
    }
}

// Event Listeners
registerBtn.addEventListener('click', registerUser);
markAttendanceBtn.addEventListener('click', captureFrame);

// Start the camera when the page loads
startCamera();
