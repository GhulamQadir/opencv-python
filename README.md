# 🖼️ **ORB Feature Detection and Image Matching**

### This project demonstrates how to use OpenCV’s ORB (Oriented FAST and Rotated BRIEF) feature detector to:

- Extract keypoints and descriptors from images
- Compare images based on their features
- Identify the most similar image in a dataset
- Visualize matches between two images

### 📂 Project Structure

.
├── images/
├── image_classifier_feature_detector.py
├── image_detection.py
├── opencv_main.ipynb
├── testing_img.jpg

## Requirements

Python 3.x
OpenCV (opencv-python)

- Install dependencies:
  pip install opencv-python

## 🚀 **How to Run**

1️⃣ Image Matching Between Two Images
File: image_detection.py
This script:

- Loads two images
- Detects ORB keypoints and descriptors
- Matches features using Brute-Force matcher
- Filters matches with Lowe’s ratio test
- Visualizes the matches in a window

#### Run it:

python image_detection.py

## 2️⃣ **Image Classifier by Feature Matching**

File: image_classifier_feature_detector.py

This script:

- Loads all images in the images folder
- Computes descriptors for each image
- Reads a test image
- Compares the test image descriptors with all stored descriptors
- Determines the best match by counting good matches
- Prints the matching image’s name if found

Run it:
python image_classifier_feature_detector.py

## **✨ How It Works**

**ORB Features:** Detect distinctive points and describe them with binary vectors.

**Brute-Force Matching:** For each descriptor in the query image, find the closest descriptors in reference images.

**Lowe’s Ratio Test:** Keep matches where the best match is clearly better than the second-best (to reduce false matches).

**Descriptor Comparison:** The image with the highest number of good matches is considered the most similar.
