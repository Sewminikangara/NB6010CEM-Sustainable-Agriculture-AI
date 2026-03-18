"""Sustainable Agriculture Advisor - web app """
import streamlit as st
import os
from PIL import Image

import config
from src.predict import DiseasePredictor

st.set_page_config(
    page_title="Sustainable Agriculture Advisor",
    page_icon="🌿",
    layout="wide"
)

st.title("Sustainable Agriculture Advisor (Plant Health Monitor)")
st.markdown("Upload a leaf image to detect plant diseases and receive treatment recommendations.")

# Sidebar: model selection
st.sidebar.title("Model Settings")
model_type = st.sidebar.selectbox("Choose Model", ["MobileNetV2", "Vision Transformer (ViT)"])
model_key = "mobilenet" if model_type == "MobileNetV2" else "vit"

weights_path = os.path.join(config.MODEL_SAVE_DIR, f"{model_key}_best.pth")
if not os.path.exists(weights_path):
    st.sidebar.warning(f"No trained weights found for {model_type}. Using random weights for demo.")
    weights_path = None

predictor = DiseasePredictor(model_type=model_key, weights_path=weights_path)

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Leaf Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        if st.button("Analyze Plant Health"):
            with st.spinner('Analyzing image...'):
                temp_path = "/tmp/temp_leaf.jpg"
                image.save(temp_path)
                result = predictor.predict(temp_path)
                st.session_state['result'] = result
                st.success("Analysis Complete!")

with col2:
    st.header("Diagnosis & Advisory")
    if 'result' in st.session_state:
        res = st.session_state['result']

        st.subheader("Result")
        if res.get('is_known', True):
            st.metric("Detected Disease", res['class'])
            st.progress(res['confidence'])
            st.write(f"**Confidence Score:** {res['confidence']:.2%}")
        else:
            st.warning("**Image Not Recognized**")
            st.write("This image does not appear to be a recognized plant leaf from our database.")
            st.write(f"**Confidence Score:** {res['confidence']:.2%} (Below threshold)")

        st.divider()
        st.subheader("Agricultural Advisory")
        st.markdown(res['advice'])
    else:
        st.info("Upload an image and click 'Analyze' to see the diagnosis.")

# Model Comparison Section
st.divider()
st.header("Model Metrics & Comparison")
comparison_path = os.path.join(config.PLOT_SAVE_DIR, "comparison.png")
if os.path.exists(comparison_path):
    st.image(comparison_path, caption="Model Performance Comparison")
else:
    st.info("Run the evaluation script to generate comparison plots.")

# Sidebar
st.sidebar.divider()
st.sidebar.subheader("Dataset Statistics")
st.sidebar.write(f"Total Classes: {config.NUM_CLASSES}")
st.sidebar.write(f"Image Resolution: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
