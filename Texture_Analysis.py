import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Load image in grayscale
image = cv2.imread("sandalwood_leaf.jpg", cv2.IMREAD_GRAYSCALE)

# Compute GLCM (Gray-Level Co-occurrence Matrix)
glcm = graycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)

# Extract texture features
contrast = graycoprops(glcm, 'contrast')[0, 0]
dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
energy = graycoprops(glcm, 'energy')[0, 0]

print(f"Contrast: {contrast}, Dissimilarity: {dissimilarity}, Homogeneity: {homogeneity}, Energy: {energy}")