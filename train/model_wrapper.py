import torch
from ultralytics import YOLO

class YOLOv8Wrapper(torch.nn.Module):
    def __init__(self, model_path="yolov8n.pt"):
        super().__init__()
        self.model = YOLO(model_path).model
        self.model.eval()  # default state

    def forward(self, x):
        return self.model(x)

def detection_loss(preds, targets):
    """
    Placeholder for a proper YOLOv8-style detection loss.
    Currently returns zero. You should replace this with:
    - classification loss
    - objectness loss
    - box regression loss
    """
    return torch.tensor(0.0, requires_grad=True).to(x.device)
