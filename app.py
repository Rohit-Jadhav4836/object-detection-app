import streamlit as st
import numpy as np
import cv2
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

from src.detector import load_model, run_detection
from src.utils import detections_to_dataframe, summarize_detections, filter_detections_by_class

st.set_page_config(page_title="Object Detection Terminal", layout="wide")

THEME_CSS = """
:root {
  --crt-green: #33ff66;
  --crt-bg: #0a0f0a;
  --crt-bg-panel: #0d130d;
}
[data-testid="stHeader"] { background: var(--crt-bg) !important; }
"""

def load_css(path: str = "assets/style.css"):
    try:
        with open(path) as f:
            base_css = f.read()
        st.markdown(f"<style>{THEME_CSS}\n{base_css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found at {path}. Using default theme.")
        st.markdown(f"<style>{THEME_CSS}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading CSS: {e}")
        st.markdown(f"<style>{THEME_CSS}</style>", unsafe_allow_html=True)

load_css()

st.markdown(
    '''<div class="terminal-header">
        <div class="terminal-dots"><span></span><span></span><span></span></div>
        <div>OBJECT_DETECTION_TERMINAL v1.0</div>
        <div></div>
    </div>''',
    unsafe_allow_html=True
)

st.markdown('<h1 class="cursor">OBJECT DETECTION APP</h1>', unsafe_allow_html=True)
st.caption("> YOLOv8 real-time object detection // image upload & camera capture_")

st.sidebar.markdown("### CONFIG")

# Model selection
model_size = st.sidebar.selectbox(
    "Model Size",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
    index=0,
    help="Larger models are more accurate but slower. n=nano (fastest), s=small, m=medium, l=large, x=extra large"
)

try:
    model = load_model(model_size)
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}")
    st.info("The model will be downloaded automatically on first run. Please ensure you have an internet connection.")
    st.stop()

# Advanced settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05,
                              help="Higher values reduce false positives. Recommended: 0.5-0.6 for better accuracy")
    iou_threshold = st.slider("IoU Threshold (NMS)", 0.1, 0.9, 0.5, 0.05,
                              help="Higher values reduce overlapping boxes. Recommended: 0.5-0.6")

    st.markdown("**Box Color Scheme:**")
    color_mode = st.radio(
        "Color boxes by:",
        ["Class (default)", "Confidence Level"],
        help="Class: Each object type gets a unique color\nConfidence: Red=low, Yellow=medium, Green=high"
    )
    color_by_confidence = (color_mode == "Confidence Level")

    if color_by_confidence:
        st.info("🔴 Red <60% | 🟡 Yellow 60-80% | 🟢 Green >80%")

# Class filtering
with st.sidebar.expander("🔍 Class Filter"):
    # Get all available classes from the model
    available_classes = sorted(model.names.values())
    selected_classes = st.multiselect(
        "Filter by classes (leave empty for all)",
        options=available_classes,
        default=[],
        help="Select specific classes to detect. Leave empty to detect all classes."
    )
    filter_enabled = len(selected_classes) > 0

st.sidebar.markdown(f"**CURRENT MODEL:** `{model_size.replace('.pt', '').upper()}`")
st.sidebar.caption("COCO dataset, 80 classes")

with st.sidebar.expander("ℹ️ Model Limitations"):
    st.markdown("""
    **COCO Dataset Constraints:**

    YOLOv8 is trained on 80 general classes:
    - ✅ person, dog, cat, bird, horse, sheep, cow, elephant, bear
    - ❌ NO specific breeds/species (leopard, jaguar, cheetah, etc.)

    **Common Misclassifications:**
    - Wild cats (leopard, cheetah) → "cat"
    - Exotic animals → closest general class
    - Small animals → may be missed or confused

    **Tips for Better Results:**
    - Increase confidence to 0.5-0.6
    - Use larger models (yolov8m/l/x) for accuracy
    - Filter to expected classes only

    For specialized detection, consider fine-tuning on a custom dataset.
    """)

mode = st.sidebar.radio("INPUT MODE", ["Image Upload", "Camera Capture"], label_visibility="visible")

st.divider()

if mode == "Image Upload":
    # Batch processing toggle
    batch_mode = st.checkbox("📦 Batch Processing Mode", help="Upload and process multiple images at once")

    if batch_mode:
        uploaded_files = st.file_uploader(
            "> UPLOAD MULTIPLE IMAGE FILES",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.info(f"📊 Processing {len(uploaded_files)} images...")

            # Process all images
            all_results = []
            progress_bar = st.progress(0)

            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    image_np = np.array(image)

                    annotated, detections, inference_time = run_detection(
                        model, image_np, conf_threshold, iou_threshold, color_by_confidence
                    )

                    if filter_enabled:
                        detections = filter_detections_by_class(detections, selected_classes)

                    all_results.append({
                        "filename": uploaded_file.name,
                        "image": image_np,
                        "annotated": annotated,
                        "detections": detections,
                        "inference_time": inference_time
                    })

                    progress_bar.progress((idx + 1) / len(uploaded_files))

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

            progress_bar.empty()

            # Display batch results
            st.success(f"✅ Processed {len(all_results)} images successfully!")

            # Summary statistics
            total_detections = sum(len(r["detections"]) for r in all_results)
            avg_inference = np.mean([r["inference_time"] for r in all_results])

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Images", len(all_results))
            col2.metric("Total Objects", total_detections)
            col3.metric("Avg Time", f"{avg_inference*1000:.0f}ms")

            # Display individual results
            st.markdown("### 📋 Batch Results")

            for result in all_results:
                with st.expander(f"📄 {result['filename']} - {len(result['detections'])} objects detected"):
                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.markdown("**Original**")
                        st.image(result['image'], use_container_width=True)

                    with col_b:
                        st.markdown("**Detected**")
                        annotated_rgb = cv2.cvtColor(result['annotated'], cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, use_container_width=True)

                    if result['detections']:
                        st.dataframe(detections_to_dataframe(result['detections']), use_container_width=True)

            # Batch export
            st.markdown("### 📥 Batch Export")
            col_exp1, col_exp2 = st.columns(2)

            with col_exp1:
                # Export all annotations as ZIP
                import io
                import zipfile

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for result in all_results:
                        annotated_pil = Image.fromarray(cv2.cvtColor(result['annotated'], cv2.COLOR_BGR2RGB))
                        img_buffer = io.BytesIO()
                        annotated_pil.save(img_buffer, format="PNG")
                        zip_file.writestr(f"detected_{result['filename']}", img_buffer.getvalue())

                st.download_button(
                    label="⬇ DOWNLOAD ALL ANNOTATED IMAGES (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="batch_detections.zip",
                    mime="application/zip"
                )

            with col_exp2:
                # Export all detections as JSON
                import json
                batch_json = {
                    "batch_info": {
                        "total_images": len(all_results),
                        "total_detections": total_detections,
                        "avg_inference_time_ms": avg_inference * 1000,
                        "model": model_size,
                        "confidence_threshold": conf_threshold,
                        "iou_threshold": iou_threshold
                    },
                    "results": [
                        {
                            "filename": r["filename"],
                            "detections": r["detections"],
                            "inference_time_ms": r["inference_time"] * 1000
                        }
                        for r in all_results
                    ]
                }

                st.download_button(
                    label="⬇ DOWNLOAD ALL DETECTION DATA (JSON)",
                    data=json.dumps(batch_json, indent=2),
                    file_name="batch_detections.json",
                    mime="application/json"
                )

        else:
            st.info("> AWAITING MULTIPLE IMAGE INPUTS...")

    else:
        # Single image mode (existing code)
        uploaded_file = st.file_uploader("> UPLOAD IMAGE FILE", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                image_np = np.array(image)

                with st.spinner("RUNNING INFERENCE..."):
                    annotated, detections, inference_time = run_detection(model, image_np, conf_threshold, iou_threshold, color_by_confidence)

                # Apply class filter if enabled
                if filter_enabled:
                    detections = filter_detections_by_class(detections, selected_classes)

                # Detection quality warnings
                if detections:
                    low_conf_count = sum(1 for d in detections if d["confidence"] < 0.6)
                    if low_conf_count > 0:
                        st.warning(f"⚠️ **Quality Alert**: {low_conf_count} detection(s) with low confidence (<60%). Consider increasing the confidence threshold to 0.6+ for more reliable results.")

                    # Check for potential overlapping detections
                    if len(detections) > len(set(d["class"] for d in detections)) * 1.5:
                        st.warning("⚠️ **Overlapping Detections**: Multiple boxes detected for the same objects. Try increasing IoU threshold to 0.6+.")

                col1, col2 = st.columns(2)
                with col1:
                    with st.container(border=True):
                        st.markdown("**> ORIGINAL**")
                        st.image(image_np, use_container_width=True)
                with col2:
                    with st.container(border=True):
                        st.markdown("**> DETECTED**")
                        # Convert BGR to RGB for correct color display
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, use_container_width=True)

                summary = summarize_detections(detections)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("OBJECTS FOUND", summary["total"])
                c2.metric("UNIQUE CLASSES", summary["unique_classes"])
                c3.metric("TOP CLASS", summary["top_class"])
                c4.metric("INFERENCE TIME", f"{inference_time*1000:.0f}ms")

                # Export functionality
                col_export1, col_export2 = st.columns(2)
                with col_export1:
                    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                    import io
                    buf = io.BytesIO()
                    annotated_pil.save(buf, format="PNG")
                    st.download_button(
                        label="⬇ DOWNLOAD ANNOTATED IMAGE",
                        data=buf.getvalue(),
                        file_name="detected_objects.png",
                        mime="image/png"
                    )
                with col_export2:
                    if detections:
                        import json
                        json_data = json.dumps(detections, indent=2)
                        st.download_button(
                            label="⬇ DOWNLOAD DETECTION DATA (JSON)",
                            data=json_data,
                            file_name="detections.json",
                            mime="application/json"
                        )

                with st.container(border=True):
                    st.markdown("**> DETECTION LOG**")
                    st.dataframe(detections_to_dataframe(detections), use_container_width=True)

            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.info("Please try uploading a different image file.")
        else:
            st.info("> AWAITING IMAGE INPUT...")

else:
    # Webcam mode with streamlit-webrtc
    st.markdown("**> LIVE WEBCAM FEED**")
    st.caption("> Grant camera permission in your browser. Detection runs frame-by-frame in real time.")

    import threading
    from streamlit_webrtc import VideoHTMLAttributes

    class YOLOVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.conf = conf_threshold
            self.iou = iou_threshold
            self.lock = threading.Lock()
            self.latest_detections = []

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            annotated, detections, _ = run_detection(model, img, self.conf, self.iou, color_by_confidence)
            with self.lock:
                self.latest_detections = detections[:]
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    ctx = webrtc_streamer(
        key="object-detection",
        video_processor_factory=YOLOVideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {"width": 640, "height": 480},
            "audio": False,
        },
        video_html_attrs=VideoHTMLAttributes(
            autoPlay=True, controls=False, muted=True,
            style={"width": "100%", "height": "auto", "border": "2px solid #33ff66", "border-radius": "4px"}
        ),
    )

    st.markdown("**> LIVE DETECTED OBJECTS**")

    @st.fragment(run_every="0.5s")
    def show_live_detections():
        if ctx.video_processor:
            with ctx.video_processor.lock:
                current_detections = ctx.video_processor.latest_detections[:]

            if filter_enabled:
                current_detections = filter_detections_by_class(current_detections, selected_classes)

            if current_detections:
                df = detections_to_dataframe(current_detections)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("> NO OBJECTS DETECTED IN CURRENT FRAME")
        else:
            st.info("> START THE WEBCAM TO SEE LIVE DETECTIONS")

    if ctx.state.playing:
        show_live_detections()
    else:
        st.info("> START THE WEBCAM TO SEE LIVE DETECTIONS")