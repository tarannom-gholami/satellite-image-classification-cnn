# Tarannom Gholami - Student ID: K12341567

from shiny.express import input, render, ui

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]


class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CNNModel()
state_dict = torch.load(
    "best_model.pth",
    map_location=DEVICE
)

_ = model.load_state_dict(state_dict)
model = model.to(DEVICE)
_ = model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])



ui.h2("Satellite Image Classification")

ui.input_file(
    "image",
    "Upload Image",
    accept=[".jpg", ".jpeg", ".png"]
)


@render.image
def uploaded_image():

    file = input.image()

    if file is None:
        return None

    return {
        "src": file[0]["datapath"],
        "width": "300px"
    }


@render.text
def prediction():

    file = input.image()

    if file is None:
        return ""

    image = Image.open(
        file[0]["datapath"]
    ).convert("RGB")

    image = transform(image)
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)

    pred = torch.argmax(probs).item()
    confidence = probs[0][pred].item()

    return f"Prediction: {class_names[pred]} | Confidence: {confidence:.2%}"


@render.table
def probabilities():

    file = input.image()

    if file is None:
        return pd.DataFrame()

    image = Image.open(
        file[0]["datapath"]
    ).convert("RGB")

    image = transform(image)
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)

    probs = probs.cpu().numpy()[0]

    return pd.DataFrame({
        "Class": class_names,
        "Probability": probs
    }).sort_values(
        "Probability",
        ascending=False
    )


