import numpy as np
import torch
import clip
from PIL import Image
import cv2


class FeatureExtractor:
    def __init__(self, device: str = "cuda: 0"):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    def extract(self, bgr_img) -> np.ndarray:
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        x = self.preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = self.model.encode_image(x).squeeze(0).cpu().numpy()
            return image_features


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeatureExtractor(device=device)

    img_path = "../database_T2/9.jpg"
    img = cv2.imread(img_path)

    feature_vector = model.extract(img)
    print(f"feature vector shape: {feature_vector.shape}")
