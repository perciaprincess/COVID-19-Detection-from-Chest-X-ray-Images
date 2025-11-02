import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)  # create structure
    model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
    model.load_state_dict(torch.load("D:/Percia_MTech/GUVI/python/Projects/covid19_detection/notebook/model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
class_names = ["COVID-19", "Normal", "Viral Pneumonia"]

# ---------------------------
# Image Preprocessing
# ---------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    return transform(image).unsqueeze(0)

# ---------------------------
# Grad-CAM Function
# ---------------------------
def grad_cam(model, img_tensor, target_layer):
    model.eval()

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()

    # Remove hooks
    handle_f.remove()
    handle_b.remove()

    # Extract stored values
    activation = activations[0][0].cpu().detach().numpy()     # shape: (C, H, W)
    gradient = gradients[0][0].cpu().detach().numpy()         # shape: (C, H, W)

    weights = gradient.mean(axis=(1,2))                       # GAP across H,W
    cam = (weights[:, None, None] * activation).sum(axis=0)   # Weighted sum

    cam = np.maximum(cam, 0)                                  # ReLU
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / cam.max()

    return cam, pred_class

def overlay_gradcam(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = np.array(image)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed

# ---------------------------
# Streamlit UI
# ---------------------------
st.markdown("""
        <div style="text-align:center; padding:25px; background:linear-gradient(135deg, #42a5f5, #7e57c2);
                    color:white; border-radius:10px; margin-bottom:20px;">
            <h2>🩺 COVID-19 X-Ray Diagnosis Assistant</h2>
        </div>
    """, unsafe_allow_html=True)
st.write("Upload a Chest X-ray Image to Predict COVID-19 / Pneumonia / Normal")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    img_tensor = preprocess_image(image)
    # For ResNet use model.layer4[-1]
    heatmap, pred_class = grad_cam(model, img_tensor, model.layer4[-1]) 

    prediction = class_names[pred_class]
    st.markdown(f"### ✅ **Prediction:** {prediction}")

    # Display Grad-CAM
    cam_image = overlay_gradcam(image, heatmap)
    st.image(cam_image, caption="Grad-CAM Lung Attention Map", use_column_width=True)