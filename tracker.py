import cv2
import numpy as np

# Load video file
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
# Select object to track
bbox = cv2.selectROI('Select Object to Track', frame)

# Initialize tracker
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)

# Loop through each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Update tracker
    success, bbox = tracker.update(frame)
    
    # Display results
    if success:
        # Draw bounding box around tracked object
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
