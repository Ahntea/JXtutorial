import torch
import cv2
import numpy as np
from monodepth2.networks import ResnetEncoder, DepthDecoder

# Load the pre-trained MonoDepth2 model
model_path = '/Users/antea/JXproject/gymnasium/mono_640x192/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = ResnetEncoder(18, False)
depth_decoder = DepthDecoder(encoder.num_ch_enc)
loaded_dict_enc = torch.load(model_path + "encoder.pth", map_location=device)
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

depth_decoder = DepthDecoder(encoder.num_ch_enc)
loaded_dict = torch.load(model_path + "depth.pth", map_location=device)
depth_decoder.load_state_dict(loaded_dict)

# Set the model to evaluation mode and move to the appropriate device
encoder.eval()
depth_decoder.eval()
encoder.to(device)
depth_decoder.to(device)

# Initialize camera capture object
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Resize the frame to the input size of the model
    resized_frame = cv2.resize(frame, (640, 192))

    # Preprocess the resized frame for input to the model
    input_data = resized_frame.transpose(2, 0, 1)
    input_data = torch.from_numpy(input_data).unsqueeze(0).float().to(device) / 255.0

    # Generate the depth map using the model's prediction
    with torch.no_grad():
        features = encoder(input_data)
        outputs = depth_decoder(features)
        depth_map = outputs[("disp", 0)].squeeze().cpu().numpy()

    # Normalize the depth map to a range of 0-255
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_norm_resize = cv2.resize(depth_norm, (640, 480))
    # Display the original frame and the depth map
    cv2.imshow('Original', frame)
    cv2.imshow('Depth Map', depth_norm_resize)

    # Check for user input to exit the program
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the camera capture object and close all windows
cap.release()
cv2.destroyAllWindows()
