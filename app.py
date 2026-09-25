from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.inference import predict, overlay_cam
from src.model import load_checkpoint
from src.rehab import load_exercises, recommendations

st.set_page_config(page_title="AI Knee Osteoarthritis Rehab System", page_icon="🦵", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }
    div[data-testid="stFileUploader"] {
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

st.session_state.setdefault("current_page", "prediction")
st.session_state.setdefault("selected_exercise_name", None)


def reset_scroll_to_top():
    st.components.v1.html(
        """
        <script>
            function resetScroll() {
                window.scrollTo(0, 0);
                document.body.scrollTop = 0;
                document.documentElement.scrollTop = 0;
            }
            setTimeout(resetScroll, 50);
            resetScroll();
        </script>
        """,
        height=0,
        scrolling=False,
    )


def render_prediction_page():
    reset_scroll_to_top()
    st.title("AI Knee OA Grading")
    st.caption("Upload an X-ray to get the predicted grade, severity, and rehab recommendation.")

    uploaded = st.file_uploader("Upload knee X-ray", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        result = predict(image, model, class_names, device)
        st.session_state["result"] = result
        st.session_state["image"] = image

    result = st.session_state.get("result")
    image = st.session_state.get("image")

    if result is not None and image is not None:
        predicted_grade = result["grade"]
        predicted_severity = result["severity"]
        confidence = result["confidence"] * 100
        exercises = recommendations(predicted_grade, load_exercises())
        exercise = exercises[0] if exercises else None

        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        with right_col:
            st.subheader("Prediction")
            st.metric("KL Grade", predicted_grade)
            st.metric("Severity", predicted_severity)
            st.metric("Confidence", f"{confidence:.1f}%")

            if exercise:
                st.write("### Recommended Rehab")
                st.write(f"**Exercise:** {exercise['name']}")
                st.write(f"**Target movement:** {exercise['target_movement']}")
                st.write(f"**Repetitions:** {exercise['repetitions']}")
                st.write(f"**Instructions:** {exercise['instructions']}")
                st.markdown("<div style='margin-top: 0.8rem;'><strong>Exercise guide:</strong> avatar-based motion preview appears in the rehabilitation screen.</div>", unsafe_allow_html=True)
            else:
                st.info("No rehab recommendation is available for this grade.")

        if exercises:
            st.write("### Exercise options")
            selected_name = st.selectbox("Choose an exercise", [item["name"] for item in exercises], index=0)
            st.session_state["selected_exercise_name"] = selected_name
            selected_exercise = next(item for item in exercises if item["name"] == selected_name)
            st.write(selected_exercise["instructions"])
            st.caption(f"Target movement: {selected_exercise['target_movement']} | Repetitions: {selected_exercise['repetitions']}")
            st.caption(selected_exercise["clinical_note"])

        st.write("---")

        if st.button("Start rehabilitation", type="primary"):
            st.session_state["current_page"] = "rehabilitation"
            st.session_state["selected_exercise_name"] = st.session_state.get("selected_exercise_name") or (exercise["name"] if exercise else None)
            st.rerun()
    else:
        st.info("Upload a knee X-ray to grade it and reveal the rehabilitation plan.")


def render_rehab_page():
    reset_scroll_to_top()
    result = st.session_state.get("result")
    if result is None:
        st.session_state["current_page"] = "prediction"
        st.rerun()

    predicted_grade = result["grade"]
    exercises = recommendations(predicted_grade, load_exercises())
    selected_name = st.session_state.get("selected_exercise_name") or (exercises[0]["name"] if exercises else None)
    selected_exercise = next((item for item in exercises if item["name"] == selected_name), exercises[0] if exercises else None)

    st.markdown(
        """
        <style>
        .rehab-screen {
            min-height: 0;
            margin-top: 0;
            padding-top: 0;
        }
        div[data-testid="stMainBlockContainer"] {
            padding-top: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="rehab-screen">', unsafe_allow_html=True)
        if st.button("← Back / Return to Prediction", key="back_to_prediction"):
            st.session_state["current_page"] = "prediction"
            st.rerun()

        st.title("Rehabilitation / Exercise Tracking")
        st.caption("Follow the exercise guidance and use the live monitor for posture and knee-angle feedback.")

        rehab_left, rehab_right = st.columns([1, 1], gap="large")

        with rehab_left:
            st.subheader("Exercise avatar")
            st.write(f"**Selected exercise:** {selected_exercise['name']}")
            st.write(f"**Movement:** {selected_exercise['target_movement']}")
            st.write(f"**Repetitions:** {selected_exercise['repetitions']}")
            st.write(selected_exercise["instructions"])

            avatar_html = """
            <div style='margin-top: 1rem; background: linear-gradient(180deg, #0f172a, #111827); border-radius: 16px; padding: 1rem; border: 1px solid rgba(255,255,255,0.12);'>
              <svg viewBox='0 0 260 260' width='100%' height='220' xmlns='http://www.w3.org/2000/svg'>
                <g stroke='white' stroke-width='7' stroke-linecap='round' fill='none'>
                  <circle cx='130' cy='38' r='24' fill='rgba(255,255,255,0.08)' stroke='#7dd3fc'/>
                  <line x1='130' y1='62' x2='130' y2='118' stroke='#93c5fd'/>
                  <line x1='130' y1='80' x2='88' y2='110' stroke='#c4b5fd'/>
                  <line x1='130' y1='80' x2='172' y2='110' stroke='#c4b5fd'/>
                  <line x1='130' y1='118' x2='98' y2='170' stroke='#f9a8d4'/>
                  <line x1='130' y1='118' x2='162' y2='170' stroke='#f9a8d4'/>
                  <line x1='98' y1='170' x2='88' y2='220' stroke='#86efac'/>
                  <line x1='162' y1='170' x2='172' y2='220' stroke='#86efac'/>
                  <g>
                    <line x1='130' y1='118' x2='110' y2='145' stroke='#fcd34d' stroke-width='9'>
                      <animateTransform attributeName='transform' type='rotate' values='0 130 118; -22 130 118; 0 130 118; 22 130 118; 0 130 118' dur='1.6s' repeatCount='indefinite'/>
                    </line>
                    <line x1='110' y1='145' x2='90' y2='170' stroke='#fcd34d' stroke-width='9'>
                      <animateTransform attributeName='transform' type='rotate' values='0 110 145; 18 110 145; 0 110 145; -18 110 145; 0 110 145' dur='1.6s' repeatCount='indefinite'/>
                    </line>
                  </g>
                </g>
              </svg>
              <div style='text-align:center; color:#bfdbfe; font-size:0.9rem; margin-top:0.3rem;'><strong>Movement cue:</strong> keep the knee aligned with the hip and ankle.</div>
            </div>
            """
            st.components.v1.html(avatar_html, height=260, scrolling=False)

        with rehab_right:
            st.subheader("Live monitor")
            try:
                import cv2
                import mediapipe as mp
                if not hasattr(mp, "solutions"):
                    st.warning("Camera monitoring needs a compatible MediaPipe install. The grade prediction still works normally.")
                else:
                    frame_slot = st.empty()
                    status_slot = st.empty()
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        pose = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.6, min_tracking_confidence=0.6)
                        try:
                            ok, frame = cap.read()
                            if ok and frame is not None:
                                frame = cv2.flip(frame, 1)
                                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                if results.pose_landmarks:
                                    from src.monitor import MonitorState, assess_exercise
                                    state = MonitorState()
                                    state = assess_exercise(results.pose_landmarks.landmark, selected_exercise["target_movement"], state)
                                    status_slot.markdown(f"<div style='padding: 0.8rem; border-radius: 12px; background: #ccfbf1; border: 1px solid #5eead4; color: #134e4a;'><strong>Feedback:</strong> {state.feedback}</div>", unsafe_allow_html=True)
                                    if state.knee_angle is not None:
                                        status_slot.markdown(f"<div style='margin-top: 0.5rem; padding: 0.8rem; border-radius: 12px; background: #dbeafe; border: 1px solid #93c5fd; color: #1e3a8a;'><strong>Knee angle:</strong> {state.knee_angle:.0f}°</div>", unsafe_allow_html=True)
                                    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                                else:
                                    status_slot.info("Pose not detected yet. Keep your full body in frame.")
                                frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                            else:
                                status_slot.error("Camera frame not available.")
                        finally:
                            cap.release()
                            pose.close()
                    else:
                        st.warning("Camera access is unavailable. The app is still working for grading and rehab recommendation.")
            except Exception:
                st.warning("Camera monitoring is unavailable in this environment. The grade prediction and rehab recommendation still work.")

        st.markdown('</div>', unsafe_allow_html=True)


if st.session_state.get("current_page") == "rehabilitation":
    render_rehab_page()
else:
    render_prediction_page()
