from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.inference import predict, overlay_cam
from src.model import load_checkpoint
from src.rehab import load_exercises, recommendations

st.set_page_config(page_title="Knee OA Assist", page_icon="K", layout="wide")
st.markdown("""
<style>
:root { --ink:#173042; --mint:#dff2e8; --coral:#ef765b; --paper:#f7f4ee; }
.stApp { background: var(--paper); color: var(--ink); }
.block-container { max-width: 1100px; padding-top: 3rem; }
.hero { padding: 2rem 0 1rem; border-bottom: 1px solid #cad8d0; margin-bottom: 1.5rem; }
.hero h1 { font-size: 2.5rem; letter-spacing: 0; margin-bottom: .25rem; }
.panel { background: white; padding: 1.2rem; border: 1px solid #d5e2da; border-radius: 8px; }
.badge { display:inline-block; background:var(--mint); padding:.35rem .65rem; border-radius:999px; font-weight:600; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><span class="badge">CLINICAL ASSISTANCE PROTOTYPE</span><h1>Knee OA grading, made understandable.</h1><p>Upload an X-ray, review the model explanation, then explore guided movement practice.</p></div>', unsafe_allow_html=True)
st.warning("Educational prototype only. It does not diagnose, replace a clinician, or prescribe exercise. Stop if you feel pain or discomfort.")

checkpoint = Path("checkpoints/best.pt")
data_root = Path("data")
dataset_ready = all((data_root / split).is_dir() for split in ("train", "test")) and (
    (data_root / "val").is_dir() or (data_root / "validation").is_dir()
)
if not checkpoint.exists():
    if dataset_ready:
        st.info("Dataset found, but no trained checkpoint exists yet. Run `python -m src.train --data-dir data --epochs 15`, then refresh this page.")
    else:
        st.info("Dataset not found at the project root. Add data/train, data/val (or data/validation), and data/test, then run the training command in README.md.")
    st.stop()

@st.cache_resource
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (*load_checkpoint(str(checkpoint), device), device)

model, class_names, device = get_model()
left, right = st.columns([1, 1])
with left:
    uploaded = st.file_uploader("Upload knee X-ray", type=["jpg", "jpeg", "png"])
with right:
    st.markdown('<div class="panel"><strong>Workflow</strong><br>1. Grade the image<br>2. Review a movement option<br>3. Watch the video<br>4. Monitor repetitions</div>', unsafe_allow_html=True)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    result = predict(image, model, class_names, device)
    st.session_state["result"] = result
    st.session_state["image"] = image
    st.subheader("Model result")
    metric_cols = st.columns(3)
    metric_cols[0].metric("KL grade", result["grade"])
    metric_cols[1].metric("Severity label", result["severity"])
    metric_cols[2].metric("Confidence", f"{result['confidence']:.1%}")
    from src.gradcam import GradCAM
    cam = GradCAM(model, model.features[-1])
    heatmap, _ = cam(result["tensor"])
    cam.close()
    st.image(overlay_cam(image, heatmap), caption="Grad-CAM: regions influencing the prediction", use_container_width=True)

if "result" in st.session_state:
    exercises = recommendations(st.session_state["result"]["grade"], load_exercises())
    st.subheader("Suggested movement options")
    selected_name = st.selectbox("Choose an exercise", [item["name"] for item in exercises])
    exercise = next(item for item in exercises if item["name"] == selected_name)
    st.write(exercise["instructions"])
    st.caption(f"Target: {exercise['target_movement']} · Suggested repetitions: {exercise['repetitions']}")
    st.link_button("Open instructional video", exercise["video_url"])
    st.caption(exercise["clinical_note"])
    st.subheader("Real-time monitoring")
    st.caption("Camera monitoring uses visible pose landmarks and transparent rule-based feedback. Ensure your full body is visible and use support where appropriate.")
    if st.button("Start camera monitoring", type="primary"):
        from src.monitor import MonitorState, assess_exercise
        import cv2
        import mediapipe as mp
        cap = cv2.VideoCapture(0)
        state = MonitorState()
        pose = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.6, min_tracking_confidence=0.6)
        drawing = mp.solutions.drawing_utils
        frame_slot = st.empty()
        status_slot = st.empty()
        for _ in range(900):
            ok, frame = cap.read()
            if not ok:
                status_slot.error("Could not read the camera.")
                break
            frame = cv2.flip(frame, 1)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                state = assess_exercise(results.pose_landmarks.landmark, exercise["target_movement"], state)
                drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            status_slot.info(f"Repetitions: {state.repetitions} | {state.feedback}")
        cap.release()
        pose.close()
