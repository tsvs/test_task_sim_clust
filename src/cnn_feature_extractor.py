import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

import cv2


class FeatureExtractor:
    def __init__(self, device: str = "cuda:0"):
        # init a pre-trained resnet
        backbone = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        layers = list(backbone.children())[:-1]
        self.tensor_transform = T.ToTensor()
        self.inference_transforms = torch.nn.Sequential(
            T.Resize((224, 224)),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )
        self.device = device
        print(f"{self.device=}")
        self.feature_extractor = nn.Sequential(*layers).to(self.device)

    def extract(self, bgr_img):
        x = self.preprocessing(bgr_img)
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        return representations.squeeze().cpu().numpy()

    def preprocessing(self, bgr_img):
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        img_tensor = self.tensor_transform(img)
        img = self.inference_transforms(img_tensor).to(self.device)
        return img.unsqueeze(0)


if __name__ == "__main__":
    model = FeatureExtractor()

    img_path = "../database_T2/9.jpg"
    img = cv2.imread(img_path)

    feature_vector = model.extract(img)
    print(f"feature vector shape: {feature_vector.shape}")
