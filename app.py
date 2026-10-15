import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

class DriverMonitoringSystem:
    def __init__(self, model_path, class_labels):
        self.model = load_model(model_path)
        self.class_labels = class_labels
        self.detector = MTCNN()

    def detect_face(self, frame):
        faces = self.detector.detect_faces(frame)

        if len(faces) > 0:
            face = faces[0]
            x, y, w, h = face['box']

            margin = 50
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            w = min(w + 2 * margin, frame.shape[1] - x)
            h = min(h + 2 * margin, frame.shape[0] - y)

            face_roi = frame[y:y+h, x:x+w]
            return face_roi, (x, y, w, h)

        return None, None

    def preprocess_image(self, face_roi):
        img = cv2.resize(face_roi, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0) / 255.0
        return img

    def predict_driver_state(self, img):
        predictions = self.model.predict(img, verbose=0)
        predicted_class = self.class_labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        return predicted_class, confidence


# Streamlit UI
st.title("Driver Monitoring System")

# Initialize session state
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

if "camera" not in st.session_state:
    st.session_state.camera = None

# Load model only once
@st.cache_resource
def load_driver_system():
    model_path = "weightedmobilenet_5class_new.keras"
    class_labels = [
        "forward",
        "mobile_usage",
        "on_phonecall",
        "rotated",
        "yawn"
    ]
    return DriverMonitoringSystem(model_path, class_labels)

system = load_driver_system()

# Start and Stop buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Start Webcam", use_container_width=True):
        if st.session_state.camera is None:
            camera = cv2.VideoCapture(0)

            if camera.isOpened():
                st.session_state.camera = camera
                st.session_state.webcam_running = True
            else:
                camera.release()
                st.error("Tidak dapat membuka kamera.")

with col2:
    if st.button("Stop Webcam", use_container_width=True):
        st.session_state.webcam_running = False

        if st.session_state.camera is not None:
            st.session_state.camera.release()
            st.session_state.camera = None

        st.rerun()

# Webcam display
FRAME_WINDOW = st.empty()

if st.session_state.webcam_running:
    cap = st.session_state.camera

    if cap is not None:
        while st.session_state.webcam_running:
            ret, frame = cap.read()

            if not ret:
                st.warning("Tidak dapat membaca frame dari kamera.")
                break

            # Detect face and predict driver state
            face_roi, box = system.detect_face(frame)

            if face_roi is not None:
                img = system.preprocess_image(face_roi)
                predicted_class, confidence = (
                    system.predict_driver_state(img)
                )

                x, y, w, h = box

                text = f"{predicted_class} ({confidence:.2f}%)"

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    frame,
                    text,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            FRAME_WINDOW.image(frame_rgb)

        # Release camera when loop ends
        if st.session_state.camera is not None:
            st.session_state.camera.release()
            st.session_state.camera = None

        st.session_state.webcam_running = False

else:
    FRAME_WINDOW.info("Webcam belum aktif.")