import torch
import torchvision.models as models

# Define the PyTorch model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set the model to evaluation mode
model.eval()

# Create some sample input
batch_size = 1
input_shape = (3, 720, 1280)
dummy_input = torch.randn(batch_size, *input_shape)

# Set the name and path of the output ONNX file
output_file = "yolov5s.onnx"

# Export the model to ONNX format
torch.onnx.export(model,
                  dummy_input,
                  output_file,
                  input_names=["input"],
                  output_names=["output"],
                  opset_version=11)


