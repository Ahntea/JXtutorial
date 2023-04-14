import cv2
import torch
import numpy as np

import matplotlib.pyplot as plt
from image_patch_min import find_minsum_patch
from image_patch_max import find_maxsum_patch

# Set up camera capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

while True:
    # Capture frame from camera
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # output = prediction.cpu().numpy().astype('uint8')
    output = prediction.cpu().numpy()
    output_color = cv2.applyColorMap(cv2.convertScaleAbs(output, alpha=255/output.max()), cv2.COLORMAP_MAGMA)

    center_x, center_y = find_minsum_patch(output,40)
    # center_x, center_y = find_maxsum_patch(output,40)
    cv2.circle(output_color,center=(center_x,center_y), radius=10, thickness=5, color=(255,255,255))
    
    cv2.imshow("Output", output_color)
    # cv2.imshow("Output", output_jet)
    # cv2.imshow('Original',frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
