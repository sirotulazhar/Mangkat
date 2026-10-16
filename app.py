
import os
import gdown
import cv2
import numpy as np
import streamlit as st
import av

from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ==========================================
# 1. DRIVER MONITORING SYSTEM
# ==========================================

class DriverMonitoringSystem:
    def __init__(self, model_path, class_labels):
        self.model = load_model(model_path)
        self.class_labels = class_labels
        self.detector = MTCNN()

    def detect_face(self, frame):
        # MTCNN menerima gambar dalam format RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(frame_rgb)

        if len(faces) > 0:
            face = faces[0]
            x, y, w, h = face["box"]

            margin = 50

            x = max(x - margin, 0)
            y = max(y - margin, 0)
            w = min(w + 2 * margin, frame.shape[1] - x)
            h = min(h + 2 * margin, frame.shape[0] - y)

            face_roi = frame[y:y + h, x:x + w]

            if face_roi.size > 0:
                return face_roi, (x, y, w, h)

        return None, None

    def preprocess_image(self, face_roi):
        img = cv2.resize(face_roi, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0) / 255.0

        return img

    def predict_driver_state(self, img):
        predictions = self.model.predict(img, verbose=0)

        predicted_class = self.class_labels[
            np.argmax(predictions)
        ]

        confidence = np.max(predictions) * 100

        return predicted_class, confidence


# ==========================================
# 2. DOWNLOAD DAN LOAD MODEL
# ==========================================

@st.cache_resource
def load_driver_system():
    model_path = "weightedmobilenet_5class_new.keras"
    temp_path = model_path + ".download"

    # ID file model di Google Drive
    file_id = "1cMr9svNXfIUd7BXwkXDYbSLgmoo9ViHj"

    # Ukuran model pada log sekitar 638 MB
    min_size = 500 * 1024 * 1024

    # Periksa apakah file model sudah lengkap
    if os.path.exists(model_path):
        if os.path.getsize(model_path) < min_size:
            os.remove(model_path)

    # Download model jika belum tersedia
    if not os.path.exists(model_path):
        if os.path.exists(temp_path):
            os.remove(temp_path)

        with st.spinner(
            "Mengunduh model dari Google Drive. Mohon tunggu..."
        ):
            result = gdown.download(
                id=file_id,
                output=temp_path,
                quiet=False
            )

        # Pastikan download berhasil dan file cukup besar
        if (
            result is None
            or not os.path.exists(temp_path)
            or os.path.getsize(temp_path) < min_size
        ):
            if os.path.exists(temp_path):
                os.remove(temp_path)

            raise RuntimeError(
                "Download model gagal atau file tidak lengkap. "
                "Periksa izin akses Google Drive."
            )

        # Simpan file setelah download selesai
        os.replace(temp_path, model_path)

    class_labels = [
        "forward",
        "mobile_usage",
        "on_phonecall",
        "rotated",
        "yawn"
    ]

    return DriverMonitoringSystem(
        model_path,
        class_labels
    )


# ==========================================
# 3. INISIALISASI MODEL
# ==========================================

st.set_page_config(
    page_title="MANGKAT - Driver Monitoring System",
    layout="wide"
)

st.title("MANGKAT - Driver Monitoring System")

st.write(
    "Sistem pemantauan kondisi pengemudi "
    "menggunakan deteksi wajah dan klasifikasi kondisi."
)

try:
    system = load_driver_system()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


# ==========================================
# 4. PEMROSESAN VIDEO WEBCAM
# ==========================================

class DriverVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.system = system

    def recv(self, frame):
        # Ambil frame dari kamera browser
        img = frame.to_ndarray(format="bgr24")

        # Deteksi wajah
        face_roi, box = self.system.detect_face(img)

        if face_roi is not None:
            # Prediksi kondisi pengemudi
            processed_img = self.system.preprocess_image(
                face_roi
            )

            predicted_class, confidence = (
                self.system.predict_driver_state(processed_img)
            )

            x, y, w, h = box

            # Gambar kotak deteksi wajah
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            # Tampilkan hasil prediksi
            text = f"{predicted_class} ({confidence:.2f}%)"

            cv2.putText(
                img,
                text,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        # Kirim kembali frame ke browser
        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ==========================================
# 5. WEBCAM UI
# ==========================================

st.subheader("Pemantauan Webcam")

st.caption(
    "Klik START untuk mengaktifkan kamera. "
    "Klik STOP untuk menghentikan pemantauan."
)

webrtc_streamer(
    key="mangkat-driver-monitoring",
    video_processor_factory=DriverVideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    async_processing=True
)
