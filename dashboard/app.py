"""
Streamlit dashboard (Section 29-31).

This file is intentionally thin -- it only handles UI/upload/display. All
actual detection logic lives in src/pipeline.py and src/video/video_processor.py
so the core system works without Streamlit (Section 31).

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import pandas as pd
import streamlit as st

from src.pipeline import RoadDamagePipeline
from src.preprocessing.image_processor import load_image_bytes, ImageProcessingError
from src.video.video_processor import VideoProcessor, VideoProcessingError
from src.database.repository import DetectionRepository
from src.detection.detector import DetectorError
from src.utils.config import get_config

st.set_page_config(page_title="Road Damage Detection System", layout="wide")


@st.cache_resource
def get_pipeline():
    return RoadDamagePipeline()


@st.cache_resource
def get_repository():
    return DetectionRepository()


def render_summary(repo: DetectionRepository):
    stats = repo.summary_stats()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total detections", stats["total_detections"])
    c2.metric("Potholes", stats["pothole_count"])
    c3.metric("Cracks", stats["crack_count"])
    c4.metric("Avg severity", stats["average_severity"])
    c5.metric("High/Critical", stats["high_critical_count"])


def render_detection_details(outcomes):
    rows = []
    for o in outcomes:
        rows.append({
            "Type": o.detection.class_name,
            "Confidence": round(o.detection.confidence, 3),
            "Size (px)": f"{o.metrics.width_px:.0f} x {o.metrics.height_px:.0f}",
            "Relative area": round(o.metrics.relative_area, 4),
            "Est. dimensions (m)": (
                f"{o.dimension.estimated_width_m} x {o.dimension.estimated_height_m}"
                if o.dimension.calibration_available else "Unavailable (uncalibrated)"
            ),
            "Severity": o.severity.severity_score,
            "Category": o.severity.severity_category,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def main():
    st.title("🛣️ Road Damage Detection System — Phase 1")
    st.caption(
        "Software MVP: pothole & crack detection with a transparent, rule-based "
        "severity score. No physical sensors are used in Phase 1 — see 'Limitations' below."
    )

    try:
        pipeline = get_pipeline()
    except DetectorError as e:
        st.error(f"Failed to load detection model: {e}")
        st.stop()

    repo = get_repository()

    st.header("Summary")
    render_summary(repo)

    st.header("Upload")
    tab_image, tab_video = st.tabs(["Image", "Video"])

    with tab_image:
        uploaded_image = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded_image is not None:
            try:
                image = load_image_bytes(uploaded_image.read(), uploaded_image.name)
            except ImageProcessingError as e:
                st.error(str(e))
                st.stop()

            with st.spinner("Running detection..."):
                outcomes, annotated, latency_ms = pipeline.process_image(image)

            st.header("Detection Result")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Inference: {latency_ms:.1f} ms", use_container_width=True)

            st.header("Detection Details")
            if outcomes:
                render_detection_details(outcomes)
                if st.button("Save to database", key="save_image"):
                    for o in outcomes:
                        repo.insert(o.to_record(source_file=uploaded_image.name, source_type="image"))
                    st.success(f"Stored {len(outcomes)} detection(s).")
                    st.rerun()
            else:
                st.info("No pothole or crack detected above the confidence threshold.")

    with tab_video:
        uploaded_video = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = Path(tmp.name)

            out_path = REPO_ROOT / "outputs" / "videos" / f"annotated_{Path(uploaded_video.name).stem}.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            progress = st.progress(0, text="Processing video...")
            frame_results_holder = []

            def on_frame(fr):
                frame_results_holder.append(fr)

            try:
                with st.spinner("Processing video (this can take a while on CPU)..."):
                    processor = VideoProcessor(pipeline)
                    frame_results = processor.process(tmp_path, output_path=out_path, on_frame=on_frame)
            except VideoProcessingError as e:
                st.error(str(e))
                st.stop()
            finally:
                progress.empty()

            total_detections = sum(len(fr.outcomes) for fr in frame_results)
            st.success(f"Processed {len(frame_results)} sampled frame(s), {total_detections} detection(s).")

            if out_path.exists():
                st.video(str(out_path))

            if st.button("Save all frame results to database", key="save_video"):
                count = 0
                for fr in frame_results:
                    for o in fr.outcomes:
                        repo.insert(o.to_record(
                            source_file=uploaded_video.name, source_type="video",
                            frame_number=fr.frame_number, video_timestamp=fr.video_timestamp_s,
                        ))
                        count += 1
                st.success(f"Stored {count} record(s).")
                st.rerun()

    st.header("History")
    rows = repo.get_all(limit=500)
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        st.dataframe(df, use_container_width=True)

        st.header("Charts")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Potholes vs Cracks")
            st.bar_chart(df["damage_type"].value_counts())
        with c2:
            st.subheader("Severity Category Distribution")
            st.bar_chart(df["severity_category"].value_counts())
    else:
        st.info("No detections stored yet.")

    with st.expander("⚠️ Limitations (Phase 1)"):
        st.markdown(
            "- No physical sensor data (accelerometer/GPS) in Phase 1.\n"
            "- Physical dimensions are approximate and require manual camera calibration "
            "in `config.yaml`; without it, only pixel measurements are shown.\n"
            "- Severity is a transparent rule/weight-based score, not learned from "
            "severity-labelled ground truth.\n"
            "- Lighting, weather, and camera angle affect detection and size estimates.\n"
            "- No real vehicle deployment has been performed in Phase 1."
        )


if __name__ == "__main__":
    main()
