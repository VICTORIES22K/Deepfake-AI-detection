import hashlib
import hmac
import io
import os
import base64
import tempfile
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

# Safe OpenCV import
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Safe MoviePy import
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy.video.io.VideoFileClip import VideoFileClip

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = b"IDHU_UNMAIYA_FORENSIC_SECRET_2026"
audit_logs_db = []

print("\n[INIT] Loading Multimodal Deepfake Models...")
MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"

try:
    # We load the exact processor to auto-resize faces to 224x224 for the ViT model
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    image_model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    image_model.eval()

    audio_classifier = pipeline("audio-classification", model="MelodyMachine/Deepfake-audio-detection-V2", device=-1)
    print(">>> SUCCESS: Models Loaded into Memory! <<<\n")
except Exception as e:
    print(f"CRITICAL ERROR LOADING MODELS: {e}")
    image_processor = None
    image_model = None
    audio_classifier = None


def generate_real_colored_heatmap(image_bytes: bytes) -> str:
    """Generates thermal compression heatmap for visual forensics."""
    try:
        orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buf = io.BytesIO()
        orig.save(buf, "JPEG", quality=90)
        buf.seek(0)
        compressed = Image.open(buf)
        diff = ImageChops.difference(orig, compressed)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema]) or 1
        scale = 255.0 / max_diff
        diff = ImageEnhance.Brightness(diff).enhance(scale * 2.0)
        
        out_buf = io.BytesIO()
        diff.save(out_buf, format="PNG")
        b64_str = base64.b64encode(out_buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"
    except Exception:
        return ""


def crop_face_if_present(pil_img):
    """Isolates the face so background noise doesn't confuse the AI model."""
    if not OPENCV_AVAILABLE:
        return pil_img

    try:
        img_np = np.array(pil_img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            margin = int(0.2 * max(w, h))
            x_min = max(0, x - margin)
            y_min = max(0, y - margin)
            x_max = min(img_np.shape[1], x + w + margin)
            y_max = min(img_np.shape[0], y + h + margin)
            
            cropped = img_np[y_min:y_max, x_min:x_max]
            return Image.fromarray(cropped)
    except Exception as e:
        print(f"[FACE CROP WARN]: {e}")
        
    return pil_img


def predict_image_probs(pil_img):
    """Computes exact probabilities directly from the model to avoid label swapping."""
    if image_model is None or image_processor is None:
        return 0.0, 1.0

    try:
        inputs = image_processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = image_model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

        id2label = image_model.config.id2label
        fake_prob = 0.0
        real_prob = 0.0

        for idx, score in enumerate(probs):
            label = str(id2label.get(idx, id2label.get(str(idx), ""))).lower()
            if "deepfake" in label or "fake" in label:
                fake_prob = float(score)
            elif "realism" in label or "real" in label:
                real_prob = float(score)

        return fake_prob, real_prob
    except Exception as e:
        print(f"[PREDICTION ERROR]: {e}")
        return 0.0, 1.0


@app.post("/api/v1/detect")
async def detect_media(file: UploadFile = File(...)):
    file_bytes = await file.read()
    sha256_hash = hashlib.sha256(file_bytes).hexdigest()
    hmac_sig = hmac.new(SECRET_KEY, sha256_hash.encode(), hashlib.sha256).hexdigest()

    content_type = file.content_type or "image/png"
    fake_prob = 0.0
    real_prob = 1.0
    heatmap_b64 = ""

    # ==========================================
    # 1. VIDEO HANDLING
    # ==========================================
    if "video" in content_type or file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(file_bytes)
            tmp_vid_path = tmp_vid.name

        v_fake_scores = []
        a_fake_score = 0.0
        audio_found = False

        try:
            clip = VideoFileClip(tmp_vid_path)
            duration = clip.duration or 0
            times = np.arange(0, duration, 1.0) 
            for t in times:
                frame_array = clip.get_frame(t)
                pil_image = Image.fromarray(frame_array)
                cropped_face = crop_face_if_present(pil_image)
                f_score, _ = predict_image_probs(cropped_face)
                v_fake_scores.append(f_score)

            if clip.audio is not None and audio_classifier:
                audio_found = True
                temp_audio_path = tempfile.mktemp(suffix=".wav")
                clip.audio.write_audiofile(temp_audio_path, logger=None)
                
                audio_preds = audio_classifier(temp_audio_path)
                for pred in audio_preds:
                    if "fake" in str(pred.get("label", "")).lower():
                        a_fake_score = float(pred.get("score", 0.0))

                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

            clip.close()
        except Exception as e:
            print(f"[VIDEO ERROR]: {e}")

        if os.path.exists(tmp_vid_path):
            os.remove(tmp_vid_path)

        avg_v_fake = sum(v_fake_scores) / len(v_fake_scores) if v_fake_scores else 0.0
        fake_prob = (avg_v_fake + a_fake_score) / 2 if audio_found else avg_v_fake
        real_prob = 1.0 - fake_prob

    # ==========================================
    # 2. AUDIO-ONLY HANDLING
    # ==========================================
    elif "audio" in content_type or file.filename.endswith(('.wav', '.mp3', '.ogg')):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_aud:
            tmp_aud.write(file_bytes)
            tmp_aud_path = tmp_aud.name

        if audio_classifier:
            preds = audio_classifier(tmp_aud_path)
            for pred in preds:
                if "fake" in str(pred.get("label", "")).lower():
                    fake_prob = float(pred.get("score", 0.0))
            real_prob = 1.0 - fake_prob

        if os.path.exists(tmp_aud_path):
            os.remove(tmp_aud_path)

    # ==========================================
    # 3. PURE IMAGE HANDLING
    # ==========================================
    else:
        try:
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            
            # Step 1: Isolate the face 
            cropped_face = crop_face_if_present(pil_img)
            
            # Step 2: Predict using raw math logits
            fake_prob, real_prob = predict_image_probs(cropped_face)
            
            print(f"\n[IMAGE ENGINE] Fake: {fake_prob:.4f} | Real: {real_prob:.4f}\n")
            heatmap_b64 = generate_real_colored_heatmap(file_bytes)
        except Exception as e:
            print(f"[IMAGE READ ERROR]: {e}")

    is_fake = fake_prob > 0.50
    trust_score = round(real_prob * 100)
    confidence = round(max(fake_prob, real_prob) * 100, 2)
    verdict = "SYNTHETIC_MEDIA" if is_fake else "REAL_MEDIA"

    audit_logs_db.insert(0, {
        "hash": sha256_hash[:16] + "...",
        "verdict": verdict,
        "trust_score": trust_score,
        "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    })

    return {
        "filename": file.filename,
        "hash": sha256_hash,
        "hmac_signature": hmac_sig,
        "payload_bytes": len(file_bytes),
        "content_type": content_type,
        "verdict": verdict,
        "trust_score": trust_score,
        "vision_analysis": {"score": round(fake_prob, 2)},
        "audio_analysis": {"score": round(fake_prob, 2) if "audio" in content_type or "video" in content_type else 0.0},
        "xai_explanation": f"Media classified as {'Synthetic AI' if is_fake else 'Authentic Real Photo'} ({confidence}% confidence).",
        "heatmap_url": heatmap_b64
    }

@app.get("/api/v1/audit-logs")
async def get_audit_logs():
    return {"logs": audit_logs_db}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)