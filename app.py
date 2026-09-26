from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.inference import predict, overlay_cam
from src.model import load_checkpoint
from src.rehab import avatar_svg, load_exercises, recommendations

st.set_page_config(page_title="AI Knee Osteoarthritis Rehab System", page_icon="🦵", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stFileUploader"] {
        margin-bottom: 1.2rem;
        border-radius: 12px;
    }
    div[data-testid="stFileUploader"] section {
        border-radius: 12px;
        border: 1px dashed rgba(56, 189, 248, 0.35);
        background: rgba(15, 23, 42, 0.4);
    }
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important;
        transform: translateY(-1px) !important;
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
                try {
                    window.parent.scrollTo(0, 0);
                } catch (error) {
                    window.scrollTo(0, 0);
                }
            }
            setTimeout(resetScroll, 50);
            resetScroll();
        </script>
        """,
        height=0,
        scrolling=False,
    )


def stop_rehab_monitor():
    pose = st.session_state.pop("rehab_pose", None)
    camera = st.session_state.pop("rehab_camera", None)
    if pose is not None:
        pose.close()
    if camera is not None:
        camera.release()
    st.session_state["monitor_active"] = False


@st.fragment(run_every=0.15)
def render_live_monitor():
    import cv2
    import mediapipe as mp

    from src.monitor import MonitorState, assess_exercise

    camera = st.session_state.get("rehab_camera")
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera.release()
            st.session_state["monitor_active"] = False
            st.error("Camera access is unavailable. Allow camera access and try again.")
            return
        st.session_state["rehab_camera"] = camera

    if not hasattr(mp, "solutions"):
        stop_rehab_monitor()
        st.error("Camera monitoring needs a compatible MediaPipe install.")
        return

    pose = st.session_state.get("rehab_pose")
    if pose is None:
        pose = mp.solutions.pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        st.session_state["rehab_pose"] = pose

    if st.button("Stop monitoring", key="stop_rehab_monitor"):
        stop_rehab_monitor()
        st.rerun()

    ok, frame = camera.read()
    if not ok or frame is None:
        st.warning("Camera frame not available. Check that the camera is connected.")
        return

    frame = cv2.flip(frame, 1)
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    monitor_state = st.session_state.get("monitor_state")
    if monitor_state is None or not hasattr(monitor_state, "correctness"):
        monitor_state = MonitorState()
    if results.pose_landmarks:
        assess_exercise(
            results.pose_landmarks.landmark,
            st.session_state["active_exercise"],
            monitor_state,
        )
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
        )
    else:
        exercise_config = st.session_state["active_exercise"]
        feedback = exercise_config.get("corrective_feedback") or {}
        monitor_state.feedback = feedback.get("no_pose", "Move into frame so your body is visible")
        monitor_state.correctness = "Not assessed"
        monitor_state.knee_angle = None
    st.session_state["monitor_state"] = monitor_state
    range_of_motion = getattr(monitor_state, "range_of_motion", None)

    video_column, feedback_column = st.columns([1.2, 1])
    with video_column:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    with feedback_column:
        st.metric("Repetitions", monitor_state.repetitions)
        st.metric("Form status", monitor_state.correctness)
        joint_angles = monitor_state.joint_angles or {}
        if joint_angles:
            for joint_name, joint_angle in joint_angles.items():
                st.metric(f"{joint_name.replace('_', ' ').title()} angle", f"{joint_angle:.0f}°")
        elif monitor_state.knee_angle is not None:
            st.metric("Knee angle", f"{monitor_state.knee_angle:.0f}°")
        else:
            st.metric("Joint angle", "--")
        if range_of_motion is not None:
            st.metric("Observed ROM", f"{range_of_motion:.0f}°")
        else:
            st.metric("Observed ROM", "--")
        if monitor_state.feedback == "Correct":
            st.success(monitor_state.feedback)
        elif monitor_state.correctness in {"Incorrect", "Not assessed"}:
            st.warning(monitor_state.feedback)
        else:
            st.info(monitor_state.feedback)


def render_prediction_page():
    reset_scroll_to_top()
    st.title("🦵 AI Knee Osteoarthritis Grading")
    st.caption("Upload a knee radiograph to analyze joint space narrowing, osteophytes, and Kellgren-Lawrence grade.")

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

        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.subheader("Uploaded Radiograph")
            st.image(image, use_container_width=True)

        with right_col:
            st.subheader("Diagnostic Prediction")

            badge_colors = {
                0: "#10b981",
                1: "#0ea5e9",
                2: "#f59e0b",
                3: "#f97316",
                4: "#ef4444",
            }
            color = badge_colors.get(predicted_grade, "#38bdf8")

            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 14px; padding: 1.4rem; margin-bottom: 1.2rem; box-shadow: 0 4px 16px rgba(0,0,0,0.25);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.9rem;">
                        <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; letter-spacing: 0.5px;">KL CLASSIFICATION</span>
                        <span style="background: {color}22; color: {color}; border: 1px solid {color}55; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 700;">
                            Grade {predicted_grade}
                        </span>
                    </div>
                    <div style="font-size: 1.9rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.3rem;">
                        {predicted_severity}
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.92rem; color: #94a3b8; margin-top: 0.6rem;">
                        <span>Confidence</span>
                        <strong style="color: #f1f5f9; font-size: 1rem;">{confidence:.1f}%</strong>
                    </div>
                    <div style="margin-top: 0.6rem; background: rgba(255,255,255,0.08); border-radius: 6px; height: 8px; overflow: hidden;">
                        <div style="background: {color}; width: {min(confidence, 100):.1f}%; height: 100%; border-radius: 6px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            if st.button("Start rehabilitation →", type="primary", use_container_width=True):
                st.session_state["current_page"] = "rehabilitation"
                st.session_state["selected_exercise_name"] = exercise["name"] if exercise else None
                st.rerun()
    else:
        st.info("Upload a knee X-ray to grade it and reveal the rehabilitation plan.")

def generate_3d_avatar_component(movement_type: str, exercise_name: str, avatar_movement: str = "") -> str:
    """WebGL 3D Avatar that animates biomechanically according to the selected exercise."""
    movement_key = avatar_movement or movement_type
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ margin: 0; overflow: hidden; background: #0b1120; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #f8fafc; }}
        #container {{ width: 100%; height: 410px; position: relative; }}
        .hud {{
          position: absolute; top: 10px; left: 12px;
          background: rgba(15, 23, 42, 0.88); backdrop-filter: blur(8px);
          padding: 8px 14px; border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.25);
          font-size: 12px; line-height: 1.5; pointer-events: none; box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }}
        .hud-title {{ font-weight: 700; color: #38bdf8; font-size: 13px; letter-spacing: 0.3px; }}
        .hud-stat {{ display: flex; gap: 14px; margin-top: 4px; font-size: 11px; color: #94a3b8; }}
        .hud-val {{ color: #f1f5f9; font-weight: 600; }}
        .controls-hint {{
          position: absolute; bottom: 8px; right: 12px;
          background: rgba(15, 23, 42, 0.75); padding: 4px 10px;
          border-radius: 6px; font-size: 10px; color: #64748b; border: 1px solid rgba(255,255,255,0.08);
        }}
      </style>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
      <div id="container">
        <div class="hud">
          <div class="hud-title">3D Avatar: {exercise_name}</div>
          <div class="hud-stat">
            <span>Target Joint: <strong class="hud-val">Knee</strong></span>
            <span>Simulated Angle: <strong id="angle-val" class="hud-val">--°</strong></span>
          </div>
          <div class="hud-stat">
            <span>Phase: <strong id="phase-val" class="hud-val" style="color: #34d399;">Active</strong></span>
          </div>
        </div>
        <div class="controls-hint">🖱️ Drag to Rotate • Scroll to Zoom</div>
      </div>
      <script>
        const container = document.getElementById('container');
        const movement = "{movement_key}";
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
        camera.position.set(2.2, 1.35, 2.5);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 0.8, 0);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        // Lighting
        scene.add(new THREE.AmbientLight(0xffffff, 0.7));
        const dirLight = new THREE.DirectionalLight(0x38bdf8, 1.2);
        dirLight.position.set(3, 4, 3);
        scene.add(dirLight);
        const fillLight = new THREE.DirectionalLight(0x818cf8, 0.6);
        fillLight.position.set(-3, 2, -2);
        scene.add(fillLight);
        scene.add(new THREE.GridHelper(6, 12, 0x38bdf8, 0x1e293b));

        // Materials
        const bodyMat = new THREE.MeshStandardMaterial({{ color: 0xcfd8dc, roughness: 0.4, metalness: 0.2 }});
        const activeJointMat = new THREE.MeshStandardMaterial({{ color: 0x10b981, emissive: 0x065f46, roughness: 0.3 }});
        const neutralJointMat = new THREE.MeshStandardMaterial({{ color: 0x64748b, roughness: 0.4 }});
        const headMat = new THREE.MeshStandardMaterial({{ color: 0x0284c7, roughness: 0.3 }});
        const visorMat = new THREE.MeshStandardMaterial({{ color: 0x38bdf8, emissive: 0x0284c7, roughness: 0.1 }});
        const propMat = new THREE.MeshStandardMaterial({{ color: 0x334155, roughness: 0.6 }});

        // Humanoid Rig Hierarchy
        const root = new THREE.Group();
        scene.add(root);

        // Pelvis
        const pelvis = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.11, 0.12, 16), bodyMat);
        pelvis.position.y = 0;
        root.add(pelvis);

        // Spine & Chest
        const spine = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.12, 0.38, 16), bodyMat);
        spine.position.y = 0.24;
        pelvis.add(spine);

        // Head & Visor
        const head = new THREE.Mesh(new THREE.SphereGeometry(0.10, 16, 16), headMat);
        head.position.y = 0.30;
        spine.add(head);
        const visor = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.04, 0.08), visorMat);
        visor.position.set(0, 0.02, 0.08);
        head.add(visor);

        // Arms (Left & Right)
        const lShoulder = new THREE.Group(); lShoulder.position.set(-0.18, 0.15, 0); spine.add(lShoulder);
        const lArm = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.035, 0.3), bodyMat);
        lArm.position.y = -0.15; lShoulder.add(lArm);

        const rShoulder = new THREE.Group(); rShoulder.position.set(0.18, 0.15, 0); spine.add(rShoulder);
        const rArm = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.035, 0.3), bodyMat);
        rArm.position.y = -0.15; rShoulder.add(rArm);

        // Left Leg
        const lHip = new THREE.Group(); lHip.position.set(-0.12, -0.05, 0); pelvis.add(lHip);
        const lThigh = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.045, 0.38), bodyMat);
        lThigh.position.y = -0.19; lHip.add(lThigh);
        const lKnee = new THREE.Group(); lKnee.position.set(0, -0.19, 0); lThigh.add(lKnee);
        const lKneeMesh = new THREE.Mesh(new THREE.SphereGeometry(0.055), neutralJointMat); lKnee.add(lKneeMesh);
        const lShin = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.038, 0.38), bodyMat);
        lShin.position.y = -0.19; lKnee.add(lShin);
        const lFoot = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.04, 0.16), bodyMat);
        lFoot.position.set(0, -0.2, 0.04); lShin.add(lFoot);

        // Right Leg (Active / Focal Leg)
        const rHip = new THREE.Group(); rHip.position.set(0.12, -0.05, 0); pelvis.add(rHip);
        const rThigh = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.045, 0.38), bodyMat);
        rThigh.position.y = -0.19; rHip.add(rThigh);
        const rKnee = new THREE.Group(); rKnee.position.set(0, -0.19, 0); rThigh.add(rKnee);
        const rKneeMesh = new THREE.Mesh(new THREE.SphereGeometry(0.062), activeJointMat); rKnee.add(rKneeMesh);
        const rShin = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.038, 0.38), bodyMat);
        rShin.position.y = -0.19; rKnee.add(rShin);
        const rFoot = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.04, 0.16), bodyMat);
        rFoot.position.set(0, -0.2, 0.04); rShin.add(rFoot);

        // Props (Chair or Support Rail depending on exercise)
        const isSeated = (movement.includes("seated") || movement === "knee_extension" || movement === "supported_sit_to_stand" || movement === "sit_to_stand");
        if (isSeated) {{
          root.position.y = 0.54;
          lHip.rotation.x = -Math.PI / 2;
          lKnee.rotation.x = Math.PI / 2;
          rHip.rotation.x = -Math.PI / 2;
          rKnee.rotation.x = Math.PI / 2;
          lShoulder.rotation.x = 0.3;
          rShoulder.rotation.x = 0.3;

          // 3D Chair
          const chairGroup = new THREE.Group();
          const seat = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.05, 0.5), propMat);
          seat.position.set(0, 0.49, -0.05); chairGroup.add(seat);
          const back = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 0.05), propMat);
          back.position.set(0, 0.74, -0.28); chairGroup.add(back);
          for (let dx of [-0.2, 0.2]) {{
            for (let dz of [-0.25, 0.15]) {{
              const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.49), propMat);
              leg.position.set(dx, 0.245, dz); chairGroup.add(leg);
            }}
          }}
          scene.add(chairGroup);
        }} else {{
          root.position.y = 0.92;
          // Support Rail in front
          const railGroup = new THREE.Group();
          const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.8), propMat);
          bar.rotation.z = Math.PI / 2; bar.position.set(0, 0.95, 0.4); railGroup.add(bar);
          const post1 = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.95), propMat);
          post1.position.set(-0.35, 0.475, 0.4); railGroup.add(post1);
          const post2 = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.95), propMat);
          post2.position.set(0.35, 0.475, 0.4); railGroup.add(post2);
          scene.add(railGroup);

          // Rest hands on rail
          lShoulder.rotation.x = -0.55;
          rShoulder.rotation.x = -0.55;
        }}

        const clock = new THREE.Clock();
        function animate() {{
          requestAnimationFrame(animate);
          const t = clock.getElapsedTime();
          controls.update();

          if (movement === "seated_knee_extension" || movement === "knee_extension") {{
            const cycle = (Math.sin(t * 1.5) + 1) / 2;
            rKnee.rotation.x = (Math.PI / 2) - (cycle * 1.42);
            const ang = Math.round(90 + cycle * 85);
            document.getElementById('angle-val').innerText = ang + '°';
            document.getElementById('phase-val').innerText = cycle > 0.88 ? "Terminal Extension (Hold)" : (Math.cos(t * 1.5) > 0 ? "Extending Knee" : "Controlled Lowering");
          }} else if (movement === "seated_heel_slide") {{
            const cycle = (Math.sin(t * 1.4) + 1) / 2;
            rHip.rotation.x = -Math.PI / 2 + (1 - cycle) * 0.22;
            rKnee.rotation.x = Math.PI / 2 + (1 - cycle) * 0.45;
            const ang = Math.round(155 - (1 - cycle) * 80);
            document.getElementById('angle-val').innerText = ang + '°';
            document.getElementById('phase-val').innerText = Math.cos(t * 1.4) > 0 ? "Sliding Forward (Extend)" : "Sliding Backward (Flex)";
          }} else if (movement === "standing_hamstring_curl") {{
            const cycle = (Math.sin(t * 1.6) + 1) / 2;
            rKnee.rotation.x = cycle * 1.55;
            const ang = Math.round(180 - cycle * 95);
            document.getElementById('angle-val').innerText = ang + '°';
            document.getElementById('phase-val').innerText = cycle > 0.88 ? "Peak Curl Hold" : (Math.cos(t * 1.6) > 0 ? "Curling Backward" : "Lowering Leg");
          }} else if (movement === "supported_sit_to_stand" || movement === "sit_to_stand") {{
            const cycle = (Math.sin(t * 1.2) + 1) / 2;
            root.position.y = 0.54 + (cycle * 0.38);
            spine.rotation.x = (1 - cycle) * 0.22;
            rHip.rotation.x = (1 - cycle) * (-Math.PI / 2);
            lHip.rotation.x = (1 - cycle) * (-Math.PI / 2);
            rKnee.rotation.x = (1 - cycle) * (Math.PI / 2);
            lKnee.rotation.x = (1 - cycle) * (Math.PI / 2);
            const ang = Math.round(90 + cycle * 85);
            document.getElementById('angle-val').innerText = ang + '°';
            document.getElementById('phase-val').innerText = cycle > 0.9 ? "Full Standing Upright" : (cycle < 0.1 ? "Seated" : (Math.cos(t * 1.2) > 0 ? "Rising (Sit-to-Stand)" : "Lowering (Stand-to-Sit)"));
          }} else if (movement === "supported_weight_shift" || movement === "weight_shift") {{
            const shift = Math.sin(t * 1.5) * 0.16;
            root.position.x = shift;
            pelvis.rotation.z = -shift * 0.4;
            rKnee.rotation.x = 0.12;
            lKnee.rotation.x = 0.12;
            document.getElementById('angle-val').innerText = '175° (Soft Knees)';
            document.getElementById('phase-val').innerText = shift > 0.04 ? "Shifting Weight Right" : (shift < -0.04 ? "Shifting Weight Left" : "Centering");
          }} else if (movement === "gentle_seated_range_of_motion") {{
            const cycle = (Math.sin(t * 1.3) + 1) / 2;
            rKnee.rotation.x = (Math.PI / 2) - (cycle * 0.75);
            const ang = Math.round(95 + cycle * 45);
            document.getElementById('angle-val').innerText = ang + '°';
            document.getElementById('phase-val').innerText = "Gentle Pain-Free Arc";
          }}

          renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
          camera.aspect = container.clientWidth / container.clientHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(container.clientWidth, container.clientHeight);
        }});
      </script>
    </body>
    </html>
    """


def render_rehab_page():
    result = st.session_state.get("result")
    if result is None:
        st.warning("No X-ray result found in this session. You can upload an X-ray on the Prediction page, or choose a KL Grade below to test rehabilitation.")
        demo_col1, demo_col2 = st.columns([1, 3])
        with demo_col1:
            grade_choice = st.selectbox("Select KL Grade for preview", options=[0, 1, 2, 3, 4], index=1)
        with demo_col2:
            st.write("")
            st.write("")
            if st.button("← Go to Prediction page to upload X-ray"):
                st.session_state["current_page"] = "prediction"
                st.session_state["monitoring_active"] = False
                st.rerun()
        predicted_grade = grade_choice
        predicted_severity = {0: "None (Healthy)", 1: "Doubtful", 2: "Minimal / Mild", 3: "Moderate", 4: "Severe"}.get(grade_choice, "Doubtful")
    else:
        predicted_grade = result["grade"]
        predicted_severity = result.get("severity", "Graded")

        if st.button("← Back to Prediction"):
            st.session_state["current_page"] = "prediction"
            st.session_state["monitoring_active"] = False
            st.rerun()

    exercises = recommendations(predicted_grade, load_exercises())
    if not exercises:
        st.error(f"No exercises configured for KL Grade {predicted_grade}.")
        return

    st.title("🦵 Knee OA Rehabilitation & Motion Analysis")
    st.info(f"**Predicted KL Grade: Grade {predicted_grade} ({predicted_severity})** — Clinically guided protocol tailored for joint unloading, ROM, and quadriceps strength.")

    # Exercise selector
    exercise_names = [e["name"] for e in exercises]
    default_index = 0
    if st.session_state.get("selected_exercise_name") in exercise_names:
        default_index = exercise_names.index(st.session_state["selected_exercise_name"])

    selected_name = st.selectbox("Recommended Exercise", exercise_names, index=default_index)
    st.session_state["selected_exercise_name"] = selected_name
    selected_exercise = next(e for e in exercises if e["name"] == selected_name)

    # Reset monitor state if user switched exercise
    if st.session_state.get("active_rehab_exercise") != selected_name:
        st.session_state["active_rehab_exercise"] = selected_name
        from src.monitor import MonitorState
        st.session_state["monitor_state"] = MonitorState()

    col1, col2 = st.columns([1.05, 1.15], gap="large")

    with col1:
        st.subheader("3D Avatar Exercise Demonstration")
        avatar_mode = st.radio("Avatar Display Mode", ["3D Animated Avatar (Interactive)", "2D Biomechanical Schematic"], horizontal=True, label_visibility="collapsed")
        
        if avatar_mode == "3D Animated Avatar (Interactive)":
            avatar_html = generate_3d_avatar_component(
                selected_exercise.get("target_movement", "knee_extension"),
                selected_exercise["name"],
                selected_exercise.get("avatar_movement", "")
            )
            st.components.v1.html(avatar_html, height=430)
        else:
            st.markdown(avatar_svg(selected_exercise), unsafe_allow_html=True)

        st.markdown(f"**Instructions:** {selected_exercise['instructions']}")
        st.markdown(f"**Clinical Note:** *{selected_exercise.get('clinical_note', 'Perform with control.')}*")
        
        info_c1, info_c2 = st.columns(2)
        with info_c1:
            st.caption(f"🎯 Target Repetitions: **{selected_exercise['repetitions']} reps**")
            st.caption(f"📐 Target Movement: **{selected_exercise.get('target_movement', 'knee_extension')}**")
        with info_c2:
            ang_range = selected_exercise.get("acceptable_angle_range", [90, 175])
            st.caption(f"📏 Angle Range: **{ang_range[0]}° - {ang_range[1]}°**")
            st.caption("🔒 Support: **Safe & Non-impact**")

    with col2:
        st.subheader("Live MediaPipe Pose Monitoring")
        if "monitor_state" not in st.session_state:
            from src.monitor import MonitorState
            st.session_state["monitor_state"] = MonitorState()

        is_active = st.session_state.get("monitoring_active", False)

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if not is_active:
                if st.button("▶ Start Exercise", type="primary", use_container_width=True, key="start_mon_btn"):
                    st.session_state["monitoring_active"] = True
                    st.rerun()
            else:
                if st.button("⏹ Stop Exercise", type="primary", use_container_width=True, key="stop_mon_btn"):
                    st.session_state["monitoring_active"] = False
                    st.rerun()

        with btn_c2:
            if st.button("🔄 Reset Counter", use_container_width=True, key="reset_mon_btn"):
                st.session_state["monitor_state"].repetitions = 0
                st.session_state["monitor_state"].phase = "ready"
                st.session_state["monitor_state"].min_angle = None
                st.session_state["monitor_state"].max_angle = None
                st.rerun()

        state = st.session_state["monitor_state"]
        m1, m2, m3, m4 = st.columns(4)
        m1_slot = m1.empty()
        m2_slot = m2.empty()
        m3_slot = m3.empty()
        m4_slot = m4.empty()
        feedback_slot = st.empty()

        def update_metrics_display(cur_state):
            m1_slot.metric("Repetitions", f"{cur_state.repetitions} / {selected_exercise['repetitions']}")
            m2_slot.metric("Knee Angle", f"{cur_state.knee_angle:.0f}°" if cur_state.knee_angle is not None else "--")
            rom_val = getattr(cur_state, "range_of_motion", None)
            m3_slot.metric("Observed ROM", f"{rom_val:.0f}°" if rom_val is not None else "--")
            m4_slot.metric("Status", cur_state.movement_status)

            if cur_state.correctness == "Correct":
                feedback_slot.success(f"✅ **Feedback:** {cur_state.feedback}")
            elif cur_state.correctness == "Incorrect":
                feedback_slot.error(f"⚠️ **Form Alert:** {cur_state.feedback}")
            elif cur_state.correctness == "In progress":
                feedback_slot.info(f"ℹ️ **Guidance:** {cur_state.feedback}")
            else:
                feedback_slot.warning(f"🔍 **Status:** {cur_state.feedback}")

        update_metrics_display(state)
        frame_slot = st.empty()

        if is_active:
            import cv2
            import mediapipe as mp
            from src.monitor import assess_exercise
            import time

            # Open hardware camera once using DirectShow for rapid, stable connection
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                pose = mp.solutions.pose.Pose(
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                try:
                    while st.session_state.get("monitoring_active", False):
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            frame_slot.warning("Webcam feed lost or camera disconnected.")
                            break

                        frame = cv2.flip(frame, 1)
                        h, w, _ = frame.shape
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = pose.process(rgb_frame)

                        if results.pose_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(
                                frame,
                                results.pose_landmarks,
                                mp.solutions.pose.POSE_CONNECTIONS,
                                mp.solutions.drawing_utils.DrawingSpec(color=(0, 230, 115), thickness=2, circle_radius=3),
                                mp.solutions.drawing_utils.DrawingSpec(color=(255, 200, 0), thickness=2, circle_radius=2)
                            )
                            state = assess_exercise(
                                results.pose_landmarks.landmark,
                                selected_exercise,
                                state
                            )
                            st.session_state["monitor_state"] = state
                            update_metrics_display(state)

                        # On-frame HUD banner overlay
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (w, 65), (15, 23, 42), -1)
                        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

                        status_color = (0, 230, 115) if state.correctness == "Correct" else ((0, 90, 240) if state.correctness == "Incorrect" else (240, 200, 50))
                        cv2.putText(frame, f"REPS: {state.repetitions}/{selected_exercise['repetitions']}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                        angle_str = f"ANGLE: {state.knee_angle:.0f} deg" if state.knee_angle is not None else "ANGLE: --"
                        cv2.putText(frame, angle_str, (180, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                        cv2.putText(frame, f"STATUS: {state.correctness.upper()}", (370, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2)
                        cv2.putText(frame, f"> {state.feedback}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 235, 255), 1)

                        frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                        time.sleep(0.01)
                finally:
                    cap.release()
                    pose.close()
            else:
                frame_slot.warning("Webcam not detected or camera is currently in use by another application.")


if st.session_state.get("current_page") == "rehabilitation":
    render_rehab_page()
else:
    render_prediction_page()
