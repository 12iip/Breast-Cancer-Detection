import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import os

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        
        self.final = nn.Conv2d(32, 1, 1)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2, 2)
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2, 2)
        e3 = self.enc3(p2)
        
        d2 = self.up2(e3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        out = torch.sigmoid(self.final(d1))
        return out

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def get_prediction(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor.to(device))
        prediction = (prediction > 0.5).float()
    return prediction

def main():
    st.title("BUSI Ultrasound Image Segmentation")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.write(f"Using device: {device}")
    
    # Model loading section
    st.sidebar.header("Model Loading")
    model_file = st.sidebar.file_uploader("Upload trained model (.pth file)", type=['pth'])
    
    model = SimpleUNet()
    if model_file is not None:
        try:
            model_bytes = model_file.read()
            model.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location=device))
            st.sidebar.success("Model loaded successfully!")
            model = model.to(device)
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
    
    # Image processing section
    st.header("Image Processing")
    
    uploaded_file = st.file_uploader("Choose an ultrasound image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None and model_file is not None:
        # Display original image
        image = Image.open(uploaded_file).convert('L')
        st.image(image, caption='Original Image', use_column_width=True)
        
        # Process image and get prediction
        image_tensor = process_image(image)
        prediction = get_prediction(model, image_tensor, device)
        
        # Convert prediction to image
        pred_image = prediction.cpu().squeeze().numpy()
        pred_pil = Image.fromarray((pred_image * 255).astype(np.uint8))
        
        # Display prediction
        st.image(pred_pil, caption='Predicted Mask', use_column_width=True)
        
        # Download options
        st.header("Download Options")
        
        # Create download buttons
        col1, col2 = st.columns(2)
        
        # Save prediction as PNG
        pred_bytes = io.BytesIO()
        pred_pil.save(pred_bytes, format='PNG')
        
        with col1:
            st.download_button(
                label="Download Mask (PNG)",
                data=pred_bytes.getvalue(),
                file_name="predicted_mask.png",
                mime="image/png"
            )
        
        # Save prediction as NumPy array
        with col2:
            np_bytes = io.BytesIO()
            np.save(np_bytes, pred_image)
            st.download_button(
                label="Download Mask (NumPy)",
                data=np_bytes.getvalue(),
                file_name="predicted_mask.npy",
                mime="application/octet-stream"
            )
        
        # Display metrics
        st.header("Image Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Image Size:", image.size)
        with col2:
            st.write("Mask Size:", pred_pil.size)
        
        # Additional options
        st.header("Visualization Options")
        threshold = st.slider("Mask Threshold", 0.0, 1.0, 0.5, 0.1)
        if threshold != 0.5:
            # Recompute prediction with new threshold
            prediction = (get_prediction(model, image_tensor, device) > threshold).float()
            pred_image = prediction.cpu().squeeze().numpy()
            pred_pil = Image.fromarray((pred_image * 255).astype(np.uint8))
            st.image(pred_pil, caption=f'Predicted Mask (Threshold: {threshold})', use_column_width=True)

if __name__ == "__main__":
    main()