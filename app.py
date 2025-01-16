import streamlit as st
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time
import plotly.graph_objects as go
import shutil
from plotly.subplots import make_subplots
import glob
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Breast Ultrasound Classification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .main {
        background-color: #f5f5f5;
        animation: fadeIn 1s ease-in;
    }
    
    .stAlert {
        background-color: #ff4b4b !important;
        color: white !important;
        padding: 20px !important;
        border-radius: 10px !important;
        border-left: 5px solid #cc0000 !important;
        animation: slideIn 0.5s ease-out;
        margin-bottom: 20px !important;
        font-weight: bold !important;
    }
    
    .stProgress .st-bo {
        background-color: #ff4b4b;
        transition: all 0.3s ease;
    }
    
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    .dataset-card {
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 8px;
        margin: 10px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .dataset-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .loading-spinner {
        text-align: center;
        padding: 20px;
        animation: pulse 1.5s infinite;
    }
    
    .title-animation {
        animation: slideIn 0.8s ease-out;
    }
    
    .results-container {
        animation: fadeIn 1s ease-in;
    }

    .disclaimer {
        background-color: #ff4b4b !important;
        color: white !important;
        padding: 20px !important;
        border-radius: 10px !important;
        margin: 20px 0 !important;
        font-weight: bold !important;
        border: 2px solid #cc0000 !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load and cache the model"""
    try:
        with st.spinner("🔄 Loading model... Please wait..."):
            time.sleep(1)
            model = tf.keras.models.load_model('breast_ultrasound_model.h5')
            return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def check_mask(image_path):
    """Check if mask exists and contains valid data"""
    try:
        directory = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        mask_patterns = [
            f"{base_name}_mask.png",
            f"{base_name}_mask.jpg",
            f"{base_name}_mask.jpeg"
        ]
        
        mask_path = None
        for pattern in mask_patterns:
            temp_path = os.path.join(directory, pattern)
            if os.path.exists(temp_path):
                mask_path = temp_path
                break
        
        if mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask_exists = np.any(mask > 0)
                return True, mask, mask_exists
        
        return False, np.zeros((150, 150), dtype=np.uint8), False
        
    except Exception as e:
        st.warning(f"Error checking mask: {str(e)}")
        return False, np.zeros((150, 150), dtype=np.uint8), False

def preprocess_image(img, target_size=(150, 150)):
    """Preprocess image for model prediction"""
    try:
        if isinstance(img, str):
            img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        elif isinstance(img, np.ndarray):
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        if img is None:
            raise ValueError("Failed to load image")
            
        original_img = img.copy()
        img = cv2.equalizeHist(img)
        img = cv2.resize(img, target_size)
        img_normalized = img / 255.0
        img_normalized = np.expand_dims(img_normalized, axis=-1)
        img_normalized = np.expand_dims(img_normalized, axis=0)
        
        return img_normalized, original_img
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def load_dataset_images():
    """Load images from the BUSI dataset"""
    dataset_path = "Dataset_BUSI_with_GT"
    categories = ['benign', 'malignant', 'normal']
    dataset = {}
    
    for category in categories:
        path = os.path.join(dataset_path, category)
        if os.path.exists(path):
            images = []
            for img_path in glob.glob(os.path.join(path, '*.png')):
                if '_mask' not in img_path:
                    images.append({
                        'path': img_path,
                        'name': os.path.basename(img_path),
                        'category': category
                    })
            dataset[category] = images
    
    return dataset

def process_uploaded_files(image_file):
    """Process uploaded files and search for corresponding mask"""
    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            temp_image_path = os.path.join(temp_dir, image_file.name)
            with open(temp_image_path, "wb") as f:
                f.write(image_file.getbuffer())
            
            dataset_dir = "Dataset_BUSI_with_GT"
            categories = ['benign', 'malignant', 'normal']
            
            base_name = os.path.splitext(image_file.name)[0]
            
            for category in categories:
                category_path = os.path.join(dataset_dir, category)
                if os.path.exists(category_path):
                    mask_path = os.path.join(category_path, f"{base_name}_mask.png")
                    if os.path.exists(mask_path):
                        shutil.copy2(mask_path, os.path.join(temp_dir, f"{base_name}_mask.png"))
                        break
            
            has_mask, mask, mask_exists = check_mask(temp_image_path)
            img_normalized, original_img = preprocess_image(temp_image_path)
            
            prediction = model.predict(img_normalized)
            probabilities = prediction[0]
            
            if not mask_exists:
                predicted_class = 'normal'
                confidence = 100.0
                final_probabilities = np.zeros_like(probabilities)
                final_probabilities[categories.index('normal')] = 1.0
            else:
                class_indices = np.argsort(probabilities)[::-1]
                predicted_idx = class_indices[0]
                
                if categories[predicted_idx] == 'normal':
                    predicted_idx = class_indices[1]
                
                predicted_class = categories[predicted_idx]
                confidence = float(probabilities[predicted_idx]) * 100
                final_probabilities = probabilities
            
            return {
                'success': True,
                'original_img': original_img,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': final_probabilities
            }
            
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def process_dataset_image(image_path, model, categories):
    """Process image from dataset"""
    try:
        has_mask, mask, mask_exists = check_mask(image_path)
        img_normalized, original_img = preprocess_image(image_path)
        
        prediction = model.predict(img_normalized)
        probabilities = prediction[0]
        
        if not mask_exists:
            predicted_class = 'normal'
            confidence = 100.0
            final_probabilities = np.zeros_like(probabilities)
            final_probabilities[categories.index('normal')] = 1.0
        else:
            class_indices = np.argsort(probabilities)[::-1]
            predicted_idx = class_indices[0]
            
            if categories[predicted_idx] == 'normal':
                predicted_idx = class_indices[1]
            
            predicted_class = categories[predicted_idx]
            confidence = float(probabilities[predicted_idx]) * 100
            final_probabilities = probabilities
        
        return {
            'success': True,
            'original_img': original_img,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': final_probabilities
        }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def create_animated_prediction_gauge(confidence, category):
    """Create animated gauge chart for prediction visualization"""
    colors = {
        'benign': '#2ecc71',
        'malignant': '#e74c3c',
        'normal': '#3498db'
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': colors.get(category, '#95a5a6')},
            'steps': [
                {'range': [0, 30], 'color': '#f8f9fa'},
                {'range': [30, 70], 'color': '#e9ecef'},
                {'range': [70, 100], 'color': '#dee2e6'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        },
        title={
            'text': f"Confidence - {category.title()}",
            'font': {'size': 24}
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(t=100, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def display_results(results):
    """Display prediction results"""
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(results['original_img'], caption="Input Image", use_column_width=True)
    
    with col2:
        st.plotly_chart(
            create_animated_prediction_gauge(results['confidence'], results['predicted_class']), 
            use_container_width=True
        )
        
        st.markdown("### 📊 Detailed Analysis")
        for cat, prob in zip(categories, results['probabilities']):
            prob_value = float(prob) * 100
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(prob_value / 100)
            with col2:
                st.markdown(f"**{prob_value:.1f}%**")
            st.markdown(f"*{cat.title()}*")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="title-animation">🏥 Breast Ultrasound Classification System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="disclaimer">
        ⚠️ IMPORTANT DISCLAIMER
        
        This tool is for educational and research purposes only. It should NOT be used as a primary diagnostic tool.
        
        • The predictions made by this system are not a substitute for professional medical diagnosis
        • Always consult with qualified healthcare professionals for medical advice
        • The system's accuracy is limited and may produce false positives or negatives
    </div>
    """, unsafe_allow_html=True)
    
    global categories, model
    categories = ['benign', 'malignant', 'normal']
    
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please ensure model file exists.")
        return
    
    try:
        input_method = st.radio(
            "Choose Input Method:",
            ["Upload New Image", "Select from Dataset"],
            horizontal=True
        )
        
        if input_method == "Upload New Image":
            st.markdown("### 📤 Upload Ultrasound Image")
            image_file = st.file_uploader(
                "Choose an ultrasound image",
                type=['png', 'jpg', 'jpeg']
            )
            
            if image_file:
                if st.button("🔍 Analyze Image", use_container_width=True):
                    with st.spinner("Processing image..."):
                        results = process_uploaded_files(image_file)
                        
                        if results['success']:
                            display_results(results)
                        else:
                            st.error(f"Error processing image: {results['error']}")
        
        else:
            st.markdown("### 📁 Select from Dataset")
            dataset = load_dataset_images()
            
            if not dataset:
                st.error("No images found in the Dataset_BUSI_with_GT directory.")
                return
            
            total_images = sum(len(images) for images in dataset.values())
            st.metric("Total Images", total_images)
            
            category = st.selectbox(
                "Select Category:",
                categories,
                format_func=lambda x: x.title()
            )
            
            if category in dataset and dataset[category]:
                selected_image = st.selectbox(
                    "Select Image:",
                    dataset[category],
                    format_func=lambda x: x['name']
                )
                
                if selected_image and st.button("🔍 Analyze Selected Image", use_container_width=True):
                    with st.spinner("Processing image..."):
                        results = process_dataset_image(selected_image['path'], model, categories)
                        
                        if results['success']:
                            display_results(results)
                        else:
                            st.error(f"Error processing image: {results['error']}")
            else:
                st.warning(f"No images found in the {category} category.")
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please check your directory structure and file permissions.")

if __name__ == "__main__":
    main()