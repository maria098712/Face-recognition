import cv2
import numpy as np
from PIL import Image

# Load the Haar cascade classifier
face_cascade = cv2.CascadeClassifier('/Users/MariaBibi/python/raw.githubusercontent.com_Mjrovai_OpenCV-Face-Recognition_master_FaceDetection_Cascades_haarcascade_frontalface_default.xml')

# Create VideoCapture object
video_cap = cv2.VideoCapture(0)

# Define the codec and frame rate for the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 20.0

# Create a VideoWriter object to save the video
video_writer = cv2.VideoWriter('output.avi', fourcc, fps, (600, 400))

while True:
    # Read the frame
    ret, video_data = video_cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("video_Live", video_data)

    # Save the frame
    video_writer.write(video_data)

    # Check if the user pressed the 'a' key
    if cv2.waitKey(10) == ord("a"):
        break

# Release the VideoCapture and VideoWriter objects
video_cap.release()
video_writer.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
