from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
import threading
import pygame
import time
import base64
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'

# Load the Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load your pre-trained face mask detection model
model = tf.keras.models.load_model("mask_detector.h5")

# Global variables to control video feed and alarm
video_active = False
lock = threading.Lock()
alarm_active = False


def detect_mask(face):
    """Detect mask on a single face."""
    resized_face = cv2.resize(face, (224, 224))  # Resize to match the model input
    normalized_face = resized_face / 255.0  # Normalize the pixel values
    input_data = np.expand_dims(normalized_face, axis=0)
    prediction = model.predict(input_data)
    label = "Mask" if prediction[0][0] > 0.5 else "No Mask"
    confidence = prediction[0][0] if label == "Mask" else 1 - prediction[0][0]
    return label, confidence


def play_alarm():
    """Play an alarm sound in a loop when triggered."""
    global alarm_active
    pygame.mixer.init()
    pygame.mixer.music.load("alert.mp3")
    pygame.mixer.music.play(-1)  # Play in a loop
    while alarm_active:
        pass  # Keep playing until alarm_active is False
    pygame.mixer.music.stop()


@app.route('/')
def index():
    """Render the home page."""
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and prediction without saving the resultant image."""
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Read the uploaded image file
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Convert the image to grayscale and detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        results = []
        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w]
            label, confidence = detect_mask(face)
            confidence_percent = confidence * 100
            confidence_text = f"{confidence_percent:.2f}%"

            # Set color and label based on mask detection
            if label == "Mask":
                color = (0, 255, 0)  # Green for with mask
            else:
                color = (0, 0, 255)  # Red for no mask

            # Draw rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # Display label and confidence score
            cv2.putText(img, f"{label} ({confidence_text})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            results.append((label, confidence_percent))

        # Encode the processed image as a base64 string
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return render_template("index.html", results=results, result_image=img_base64)
    return redirect(url_for('index'))



@app.route('/start_video')
def start_video():
    """Start the live video feed."""
    global video_active
    with lock:
        video_active = True
    return redirect(url_for('index'))


@app.route('/stop_video')
def stop_video():
    """Stop the live video feed."""
    global video_active
    with lock:
        video_active = False
    return redirect(url_for('index'))


def generate_frames():
    """Generate frames for live video feed."""
    global video_active, alarm_active
    cap = cv2.VideoCapture(0)
    while True:
        with lock:
            if not video_active:
                break

        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        face_detected_without_mask = False

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            label, confidence = detect_mask(face)

            # Convert confidence to percentage format
            confidence_percent = confidence * 100
            confidence_text = f"{confidence_percent:.2f}%"

            # Set color and label based on mask detection
            if label == "Mask":
                color = (0, 255, 0)  # Green for with mask
            else:
                color = (0, 0, 255)  # Red for no mask

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Display label and confidence score
            cv2.putText(frame, f"{label} ({confidence_text})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # If no mask detected, activate alarm
            if label == "No Mask":
                face_detected_without_mask = True

        # If face detected without a mask, trigger alarm
        if face_detected_without_mask and not alarm_active:
            alarm_active = True
            threading.Thread(target=play_alarm, daemon=True).start()

        # If no face without a mask, stop alarm
        elif not face_detected_without_mask and alarm_active:
            alarm_active = False

        # Encode the frame to be sent to the client
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    alarm_active = False





@app.route('/video_feed')
def video_feed():
    """Route for live video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
