import cv2
import torch
import torchvision
import numpy as np
import time
import onnxruntime as ort

# Define the pre-trained Faster R-CNN model
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)



cap = cv2.VideoCapture(1)
while True:
    _, image = cap.read()

    # Load the input image
    session = ort.InferenceSession("/Users/antea/JXproject/gymnasium/yolov5s.onnx")
    # Convert the image to a PyTorch tensor
    # image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    resized_image = cv2.resize(image, (640,640))
    height, width, channels = resized_image.shape
    normalized_image = resized_image.astype(np.float32) / 255.0
    input_data = np.expand_dims(normalized_image.transpose(2, 0, 1), axis=0)
    # Perform the object detection inference
    # Perform inference on the input image
    output = session.run(output_names, {input_name: input_data})
    print(len(output[0][0]),output[0][0][1].shape)
    class_ids = []
    confidences = []
    boxes = []
    for out in output:
        for temp in out:
            for detection in temp:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = detection[4]
                
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # 좌표
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # Draw the bounding boxes on the input image
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            # label = str(classes[class_ids[i]])
            label = str(class_ids[i])
            color = (0,255,0)
            cv2.rectangle(image, (x, y), (x + w, y + h),color= color,thickness= 2)
            cv2.putText(image, str(label)+str(confidences[i]), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the output image
    cv2.imshow('Output Image', image)
    cv2.waitKey(1)