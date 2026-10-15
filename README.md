# MANGKAT - Driver Monitoring System

## Requirements

- Python 3.10 or 3.11 recommended for the pinned TensorFlow version.
- The trained model file: `weightedmobilenet_5class_new.keras`.

## Notes

- The five labels are retained: `forward`, `mobile_usage`, `on_phonecall`, `rotated`, `yawn`.
- The pipeline resizes each face crop to 224×224, keeps RGB input, and normalizes pixels to 0–1.
- MTCNN receives RGB frames. This is intentional because MTCNN expects RGB image data.
- The app uses the highest-confidence detected face, matching the original intent of selecting one face.
- Real-time performance depends on your computer and model size. If inference is slow, lower the camera resolution/frame rate or process fewer frames.
- Test the model output shape and label order against the training code before relying on predictions.
- For public deployment, verify package compatibility and hosting memory/CPU limits. Webcam access requires browser permission and a secure context (localhost works for local testing; deployed sites should use HTTPS).
