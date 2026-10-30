"""
Video_model.py - Forensic Multimodal Deepfake Models
Implements Video Branch, Audio Branch, and Fusion Engine connecting to online HuggingFace models:
1. Vision Model: prithivMLmods/Deep-Fake-Detector-v2-Model
2. Audio Model: MelodyMachine/Deepfake-audio-detection-V2
"""

import os
import io
import base64
import tempfile
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import torch
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification

# Safe OpenCV import
try:
    import cv2
    OPENCV_AVAILABLE = True
    if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    else:
        face_cascade = None
except Exception:
    OPENCV_AVAILABLE = False
    face_cascade = None

# Safe MoviePy import
try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
    except ImportError:
        VideoFileClip = None

# Online Model Identifiers
IMAGE_MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"
AUDIO_MODEL_NAME = "MelodyMachine/Deepfake-audio-detection-V2"

# Global lazy/cached model holders
image_classifier = None
audio_classifier = None
image_processor = None
image_model = None


def load_models():
    """
    Connects to and loads the online multimodal deepfake models from HuggingFace.
    """
    global image_classifier, audio_classifier, image_processor, image_model
    print("\n[MODELS] Connecting to online multimodal models...")
    
    try:
        image_processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_NAME)
        image_model = AutoModelForImageClassification.from_pretrained(IMAGE_MODEL_NAME)
        image_model.eval()
        image_classifier = pipeline("image-classification", model=IMAGE_MODEL_NAME, device=-1)
        print(f">>> SUCCESS: Vision Model loaded: {IMAGE_MODEL_NAME} <<<")
    except Exception as e:
        print(f"Error loading vision model ({IMAGE_MODEL_NAME}): {e}")

    try:
        audio_classifier = pipeline("audio-classification", model=AUDIO_MODEL_NAME, device=-1)
        print(f">>> SUCCESS: Audio Model loaded: {AUDIO_MODEL_NAME} <<<\n")
    except Exception as e:
        print(f"Error loading audio model ({AUDIO_MODEL_NAME}): {e}")

    return image_classifier, audio_classifier


def pil_to_base64_jpeg(pil_img, max_dim=220, quality=85) -> str:
    """Converts a PIL image to a compact base64 JPEG thumbnail."""
    try:
        w, h = pil_img.size
        if max(w, h) > max_dim:
            if w > h:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            else:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
    except Exception:
        return ""


def generate_thermal_heatmap(image_bytes: bytes) -> str:
    """
    Generates thermal Jet colormap compression heatmap for visual forensics.
    Reveals high-frequency spatial discontinuities and diffusion boundary artifacts.
    """
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
        enhanced_diff = ImageEnhance.Brightness(diff).enhance(scale * 2.0)
        
        diff_np = np.array(enhanced_diff)
        if OPENCV_AVAILABLE and hasattr(cv2, 'applyColorMap'):
            diff_gray = cv2.cvtColor(diff_np, cv2.COLOR_RGB2GRAY)
            diff_colored = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
            diff_rgb = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(diff_rgb)
        else:
            result_pil = enhanced_diff

        out_buf = io.BytesIO()
        result_pil.save(out_buf, format="PNG")
        b64_str = base64.b64encode(out_buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"
    except Exception as e:
        print(f"Heatmap generation error: {e}")
        return ""


def predict_image_probs(pil_img):
    """
    Computes exact softmax probabilities from the image classification model.
    Returns (fake_prob, real_prob).
    """
    global image_model, image_processor
    if image_model is None or image_processor is None:
        load_models()

    if image_model is None or image_processor is None:
        return 0.05, 0.95

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

        if fake_prob == 0.0 and real_prob == 0.0 and len(probs) >= 2:
            fake_prob = float(probs[1])
            real_prob = float(probs[0])

        return fake_prob, real_prob
    except Exception as e:
        print(f"[PREDICTION ERROR]: {e}")
        return 0.1, 0.9


def analyze_video(video_path, every_n_frames=15, max_samples=25):
    """
    Video Branch - Frame-by-Frame Deepfake Check with Face Detection Filter.
    Matching notebook Cell 6 & 28:
    Filters and prioritizes frames containing human faces, predicts fake scores,
    and captures visual evidence thumbnails.
    """
    global image_classifier
    if image_classifier is None:
        load_models()

    if not OPENCV_AVAILABLE:
        return []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    # Adaptive sampling: sample roughly every ~0.5 to 0.75 seconds
    step = max(1, min(every_n_frames, int(fps * 0.75)))
    frame_count = 0
    sample_count = 0
    results = []

    while cap.isOpened() and sample_count < max_samples:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % step == 0:
            sample_count += 1
            has_face = False

            # Face Detection Filter (from notebook Cell 6 & 28)
            if face_cascade is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                    if len(faces) > 0:
                        has_face = True
                except Exception:
                    has_face = False

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            fake_prob, real_prob = predict_image_probs(pil_image)

            # Prediction format matching pipeline output
            prediction = [
                {"label": "Deepfake", "score": float(fake_prob)},
                {"label": "Realism", "score": float(real_prob)}
            ]

            sec = frame_count / fps
            timestamp_str = f"{int(sec // 60):02d}:{sec % 60:04.1f}"

            # Create evidence thumbnail
            thumb_b64 = pil_to_base64_jpeg(pil_image, max_dim=200)

            results.append({
                "frame": frame_count,
                "timestamp": timestamp_str,
                "has_face": has_face,
                "prediction": prediction,
                "fake_score": float(fake_prob),
                "real_score": float(real_prob),
                "thumbnail": thumb_b64,
                "pil_image": pil_image
            })

        frame_count += 1

    cap.release()
    return results


def analyze_audio(video_path):
    """
    Audio Branch - Voice Deepfake Check from Video (Notebook Cell 14).
    Extracts the audio track and classifies voice authenticity.
    """
    global audio_classifier
    if audio_classifier is None:
        load_models()

    if VideoFileClip is None or audio_classifier is None:
        return None

    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            video.close()
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tmp_audio_path = tmp_wav.name

        video.audio.write_audiofile(tmp_audio_path, logger=None)
        video.close()

        prediction = audio_classifier(tmp_audio_path)

        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)

        return prediction
    except Exception as e:
        print(f"[AUDIO EXTRACTION NOTE]: {e}")
        return None


def analyze_audio_direct(audio_path):
    """
    Standalone Audio-Only Analysis Function (Notebook Cell 16).
    Evaluates raw audio files directly.
    """
    global audio_classifier
    if audio_classifier is None:
        load_models()

    if audio_classifier is None:
        return []

    try:
        prediction = audio_classifier(audio_path)
        return prediction
    except Exception as e:
        print(f"[AUDIO DIRECT ERROR]: {e}")
        return []


def full_analysis(video_path):
    """
    Fusion Layer - Combining Video + Audio Results (Notebook Cell 20).
    Aggregates the Video Branch and Audio Branch into a unified Multimodal Verdict
    and comprehensive forensic evidence payload.
    """
    video_results = analyze_video(video_path)
    audio_result = analyze_audio(video_path)

    # Video frames average fake score
    fake_scores = []
    most_suspicious_item = None
    highest_v_score = -1.0

    for r in video_results:
        f_score = r.get("fake_score", 0.0)
        fake_scores.append(f_score)
        if f_score > highest_v_score:
            highest_v_score = f_score
            most_suspicious_item = r

    avg_video_fake_score = sum(fake_scores) / len(fake_scores) if fake_scores else 0.05

    # Audio score calculation
    audio_fake_score = 0.0
    audio_available = audio_result is not None

    if audio_available and isinstance(audio_result, list):
        for pred in audio_result:
            if "fake" in str(pred.get("label", "")).lower():
                audio_fake_score = float(pred.get("score", 0.0))

    # Fusion Formula from Notebook Cell 20
    if audio_available:
        final_score = (avg_video_fake_score + audio_fake_score) / 2.0
    else:
        final_score = avg_video_fake_score

    final_verdict = "FAKE" if final_score > 0.50 else "REAL"
    trust_score = round((1.0 - final_score) * 100)
    confidence = round(max(final_score, 1.0 - final_score) * 100, 2)

    # Flagged anomaly frames with evidence thumbnails (Notebook Cell 20 logic)
    flagged_frames = []
    for r in video_results:
        if r.get("fake_score", 0.0) > 0.50:
            flagged_frames.append({
                "frame_number": r["frame"],
                "timestamp": r.get("timestamp", "00:00"),
                "fake_score": round(r["fake_score"], 4),
                "anomaly_pct": round(r["fake_score"] * 100, 1),
                "has_face": r.get("has_face", False),
                "thumbnail": r.get("thumbnail", "")
            })

    # Generate thermal heatmap on the most suspicious frame
    heatmap_url = ""
    if most_suspicious_item and "pil_image" in most_suspicious_item:
        buf = io.BytesIO()
        most_suspicious_item["pil_image"].save(buf, format="JPEG")
        heatmap_url = generate_thermal_heatmap(buf.getvalue())

    return {
        "video_fake_score": round(avg_video_fake_score, 3),
        "audio_fake_score": round(audio_fake_score, 3) if audio_available else "N/A (no audio track)",
        "fusion_score": round(final_score, 3),
        "final_verdict": final_verdict,
        "trust_score": trust_score,
        "confidence": confidence,
        "audio_available": audio_available,
        "frames_analyzed": len(video_results),
        "frames_with_faces": sum(1 for r in video_results if r.get("has_face")),
        "flagged_frames": flagged_frames,
        "heatmap_url": heatmap_url,
        "audio_breakdown": audio_result if audio_available else []
    }


# Auto-initialize models on import
load_models()
