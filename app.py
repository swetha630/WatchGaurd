import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
from collections import Counter
import tempfile

# ------------------ DEVICE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ LOAD MODEL ------------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4)

model.load_state_dict(torch.load("resnet_model.pth", map_location=device))
model = model.to(device)
model.eval()

class_names = ['anomaly', 'masked', 'normal', 'theft']

# ------------------ TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------ IMAGE PREDICTION ------------------
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
    return class_names[pred.item()]

# ------------------ VIDEO PREDICTION ------------------
def predict_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        label = predict_image(image)
        predictions.append(label)

    cap.release()

    final_pred = Counter(predictions).most_common(1)[0][0]
    return final_pred

# ------------------ UI ------------------
st.title("🔍 Surveillance Detection App")

option = st.radio("Choose input type:", ["Image", "Video"])

if option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        result = predict_image(image)
        st.success(f"Prediction: {result}")

elif option == "Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

    if uploaded_file:
        st.video(uploaded_file)

        result = predict_video(uploaded_file)
        st.success(f"Final Video Prediction: {result}")
