# Sandalwood-Biomachines
OpenCV-Based Models for Sandalwood Spike Disease (SSD) Detection 

1. Feature Extraction
To detect SSD based on physical symptoms like witch’s broom branching and leaf discoloration.

- Color Segmentation:
> Convert images to HSV color space.
> Identify yellowing or reddish discoloration in leaves.
> cv2.inRange() to isolate affected regions.

- Edge Detection & Shape Analysis:
> cv2.Canny() to detect sharp, abnormal branching.
> cv2.findContours() to analyze leaf and branch deformation.

- Texture Analysis:
> Use GLCM (Gray-Level Co-occurrence Matrix) or LBP (Local Binary Pattern) for feature extraction.
> Apply a Support Vector Machine (SVM) classifier for detection.


2. Deep Learning 
Deep learning model with OpenCV for real-time detection.

- Pretrained Models:
> MobileNetV2 (Lightweight, good for drones)
> YOLOv8 (You Only Look Once) (Best for real-time object detection)

3. Hyperspectral Imaging 
Uses infrared cameras to detect chlorophyll degradation. Only usable if you integrate it with a multispectral camera on an agri-drone.
- OpenCV’s PCA (Principal Component Analysis) can help in analyzing spectral signatures.

Workflow:
Collect & Label images (Healthy vs. Infected)
Train a CNN model (TensorFlow/Keras or PyTorch)
Convert model to OpenCV .pb (for TensorFlow) or .onnx (for PyTorch)
Deploy on edge devices (Jetson Nano, Raspberry Pi) or integrate into drone cameras.
