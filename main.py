import torch
from efficientnet_pytorch import EfficientNet
from torch import nn
from torchvision import models

from Utils.helpers import classification_and_detection, detection_with_image_upload


if __name__ == "__main__":
    # Load the pre-trained model state_dict
    num_classes = 83
    model_path = 'models/efficientnet_test4.pt'
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    # Usage of EfficientNet
    model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
    model.load_state_dict(state_dict)

    # Uncomment for Usage of resnet50
    # model = models.resnet50(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    # model.load_state_dict(state_dict)

    model.eval()

    # classification_and_detection(model)
    detection_with_image_upload("dataset/user_uploads/arnold.jpeg")
