import numpy as np
import cv2
import torch
from torchvision import transforms

import sys
sys.path.append('/models')
from lmdetection.models.pfld import PFLDInference, AuxiliaryNet




class LMDetector:
    def __init__(self, device, model_path="lmdetection/saved/checkpoint_epoch_313.pth.tar"):
        self.device = device
        checkpoint = torch.load(model_path,  map_location=device)
        self.model = PFLDInference().to(device)
        self.model.load_state_dict(checkpoint['pfld_backbone'])
        self.model.eval()
        self.model = self.model.to(self.device)

        self.transform = transforms.Compose([transforms.ToTensor()])


    def detection(self, img, det):
       
        img_h, img_w = img.shape[:2]
        x1, y1, x2, y2 = (det[:4] + 0.5).astype(np.int32)

        w = x2 - x1 + 1
        h = y2 - y1 + 1
        cx = x1 + w // 2
        cy = y1 + h // 2

        size = int(max([w, h]) * 1.1)
        x1 = cx - size // 2
        x2 = x1 + size
        y1 = cy - size // 2
        y2 = y1 + size

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        edx1 = max(0, -x1)
        edy1 = max(0, -y1)
        edx2 = max(0, x2 - img_w)
        edy2 = max(0, y2 - img_h)

        cropped = img[y1:y2, x1:x2]
        if (edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0):
            cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2,
                                            cv2.BORDER_CONSTANT, 0)

        input = cv2.resize(cropped, (112, 112))
        input = self.transform(input).unsqueeze(0).to(self.device)
        _, landmarks = self.model(input)
        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(
            -1, 2) * [size, size] - [edx1, edy1]
        
        result = []
        for p in pre_landmark:
            x = p[0] + x1 
            y = p[1] + y1
            result.append([int(x), int(y)])

        return result