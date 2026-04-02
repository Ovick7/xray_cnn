import os
import sys
import argparse
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from model import ChestXRayCNN

IMG_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path):
    model = ChestXRayCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(model, image_path):
    tensor = preprocess_image(image_path).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    predicted_idx = np.argmax(probs)
    return CLASS_NAMES[predicted_idx], probs[predicted_idx], probs


def main():
    parser = argparse.ArgumentParser(description="Chest X-Ray Pneumonia Classifier")
    parser.add_argument("image_path", type=str, help="Path to the X-ray image")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Path to model checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    model = load_model(args.checkpoint)
    predicted_class, confidence, probs = predict(model, args.image_path)

    print(f"Image         : {args.image_path}")
    print(f"Prediction    : {predicted_class}")
    print(f"Confidence    : {confidence:.4f} ({confidence * 100:.2f}%)")
    print(f"NORMAL prob   : {probs[0]:.4f}")
    print(f"PNEUMONIA prob: {probs[1]:.4f}")


if __name__ == "__main__":
    main()
