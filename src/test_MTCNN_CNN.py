#!/usr/bin/env python3
"""
Filename: test_MTCNN_CNN.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script leverages the MTCNN for real-time face detection and a custom CNN model for emotion recognition from video streams.
             It features live emotion classification, the ability to record segments of the video, and real-time tracking of facial features using OpenCV's KCF tracker.
Date: 2025-04-09
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from facenet_pytorch import MTCNN
from statistics import mode
from datetime import datetime, timedelta

def preprocess_input(images):
    """
    Normalize facial data by mapping pixel values from 0-255 to 0-1.
    
    Args:
        images (numpy.ndarray): Input image array with pixel values in range [0, 255]
        
    Returns:
        numpy.ndarray: Normalized image array with pixel values in range [0, 1]
    """
    return images / 255.0

def gaussian_weights_init(m):
    """
    Initialize weights for convolutional layers with Gaussian distribution.
    
    Args:
        m (nn.Module): PyTorch module to initialize weights for
    """
    if 'Conv' in m.__class__.__name__:
        m.weight.data.normal_(0.0, 0.04)

class FaceCNN(nn.Module):
    """
    Custom CNN model for emotion recognition.
    The architecture consists of:
    - Three convolutional blocks with batch normalization and max pooling
    - Four fully connected layers with dropout for regularization
    """
    def __init__(self):
        super(FaceCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Initialize weights for convolutional layers
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),  # Dropout for regularization
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.5),  # Higher dropout for regularization
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256, out_features=7),  # Output layer for 7 emotions
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 7)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)  # Flatten the features
        y = self.fc(x)
        return y

def setup_models_and_tracker():
    """
    Initialize all required models and trackers:
    - MTCNN for face detection
    - CNN model for emotion classification
    - KCF tracker for face tracking
    - Emotion labels mapping
    
    Returns:
        tuple: (mtcnn, emotion_classifier, emotion_labels, tracker)
    """
    mtcnn = MTCNN(keep_all=True, device='cpu')  # Initialize MTCNN face detector
    emotion_classifier = torch.load('./model/model_cnn.pkl', map_location=torch.device('cpu'))  # Load trained model
    emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    tracker = cv2.TrackerKCF_create()  # Initialize KCF tracker
    return mtcnn, emotion_classifier, emotion_labels, tracker

def initialize_video_capture():
    """
    Setup video capture for webcam and create display window.
    
    Returns:
        cv2.VideoCapture: Video capture object
    """
    video_capture = cv2.VideoCapture(0)  # Open default camera
    cv2.namedWindow('window_frame')  # Create display window
    return video_capture

def process_frame(frame, bbox, tracker, mtcnn, emotion_classifier, emotion_labels):
    """
    Process each frame for face detection, tracking, and emotion classification.
    
    Args:
        frame (numpy.ndarray): Input video frame
        bbox (tuple): Current bounding box coordinates (x, y, w, h)
        tracker (cv2.Tracker): KCF tracker object
        mtcnn (MTCNN): Face detection model
        emotion_classifier (nn.Module): Emotion classification model
        emotion_labels (dict): Mapping of emotion indices to labels
        
    Returns:
        tuple: (processed_frame, updated_bbox)
    """
    frame = cv2.flip(frame, 1)  # Flip frame horizontally for mirror effect
    if bbox is not None:
        # Use KCF Tracker if face was previously detected
        ok, bbox = tracker.update(frame)
        if ok:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (84, 255, 159), 2)
    else:
        # Detect faces if no face is being tracked
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None and len(boxes) > 0:
            bbox = boxes[0]
            x, y, x2, y2 = map(int, bbox)
            w, h = x2 - x, y2 - y
            bbox = (x, y, w, h)
            tracker.init(frame, bbox)

    # Process face for emotion recognition
    if bbox is not None and w > 0 and h > 0:
        face = frame[y:y + h, x:x + w]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
        face_preprocessed = preprocess_input(np.array(face_resized, dtype=np.float32)).reshape(1, 1, 48, 48)
        face_tensor = torch.from_numpy(face_preprocessed).type(torch.FloatTensor)

        # Predict emotion
        with torch.no_grad():
            emotion_pred = emotion_classifier(face_tensor)
            emotion_arg = torch.argmax(emotion_pred).item()
            emotion = emotion_labels[emotion_arg]
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame, bbox

def main():
    """
    Main function that:
    1. Initializes models and video capture
    2. Processes video frames in real-time
    3. Handles video recording and screenshots
    4. Manages user interaction
    """
    # Initialize models and video capture
    mtcnn, emotion_classifier, emotion_labels, tracker = setup_models_and_tracker()
    video_capture = initialize_video_capture()
    
    # Initialize variables for video recording
    record_flag = False
    video_writer = None
    start_time = None
    bbox = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Process frame for face detection and emotion recognition
        frame, bbox = process_frame(frame, bbox, tracker, mtcnn, emotion_classifier, emotion_labels)

        # Handle video recording
        if record_flag:
            if video_writer is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_writer = cv2.VideoWriter(f'video_{timestamp}.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10.0, (frame.shape[1], frame.shape[0]))
                start_time = datetime.now()
            video_writer.write(frame)
            if datetime.now() - start_time >= timedelta(seconds=5):  # Record for 5 seconds
                record_flag = False
                video_writer.release()
                video_writer = None

        # Display frame and handle user input
        cv2.imshow('window_frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):  # Quit
            break
        elif key == ord('r'):  # Start recording
            record_flag = True
        elif key == ord('s'):  # Take screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f'screenshot_{timestamp}.png', frame)

    # Cleanup
    video_capture.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
