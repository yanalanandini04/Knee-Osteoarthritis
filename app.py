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

st.title("AI Knee OA Grading")
st.caption("Upload an X-ray to get the predicted grade, severity, and rehab recommendation.")

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
            st.link_button("Open video", exercise["video_url"])
        else:
            st.info("No rehab recommendation is available for this grade.")

    if exercises:
        st.write("### Exercise options")
        selected_name = st.selectbox("Choose an exercise", [item["name"] for item in exercises])
        selected_exercise = next(item for item in exercises if item["name"] == selected_name)
        st.write(selected_exercise["instructions"])
        st.caption(f"Target movement: {selected_exercise['target_movement']} | Repetitions: {selected_exercise['repetitions']}")
        st.caption(selected_exercise["clinical_note"])

else:
    st.info("Upload a knee X-ray to grade it and reveal the rehabilitation plan.")
