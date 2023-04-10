import cv2
import torch
import torchvision
import numpy as np
import time

# Define the pre-trained Faster R-CNN model
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
FPS = 60
prev_time = 0


# Set the model to evaluation mode
model.eval()

def findFace(img):
    faceCascade = cv2.CascadeClassifier("/Users/antea/JXproject/gymnasium/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,255),2)

cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()

    # findFace(img)
    # cv2.imshow("Output", img)
    # cv2.waitKey(1)

    # Load the input image
    image = img
    # Convert the image to a PyTorch tensor
    # image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    image_tensor = [img]
    # print(image.shape)
    # Perform the object detection inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract the bounding boxes and class labels of the detected objects
    
    # boxes = predictions[0]['boxes'].cpu().numpy().astype('int')
    # labels = predictions[0]['labels'].cpu().numpy().astype('int')
    pred = predictions.pandas().xyxy[0]
    boxes = []
    labels = pred['name']
    confidence = pred['confidence']
    for box in zip(pred['xmin'].astype('int'),pred['ymin'].astype('int'),pred['xmax'].astype('int'),pred['ymax'].astype('int')):
        boxes.append(box)

    # Draw the bounding boxes on the image
    for box, label in zip(boxes, labels):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.circle(image,center=((box[0]+box[2])//2,(box[1]+box[3])//2), radius=2, thickness=2, color=(255,0,0))
        cv2.putText(image, str(label)+str(confidence[0]), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    current_time = time.time() - prev_time

    if current_time > 1./ FPS :
    	
        prev_time = time.time()
    # Display the output image
    cv2.imshow('Output Image', image)
    cv2.waitKey(1)