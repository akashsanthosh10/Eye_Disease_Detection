import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_clahe(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    
    # Merge back the channels
    enhanced_lab = cv2.merge([l, a, b])
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

# Example usage
image_path = "dataset\cataract\_1_5346540.jpg"  # Ensure this path is correct
enhanced_image = apply_clahe(image_path)

# Read the original image
original_image = cv2.imread(image_path)

# Display original and enhanced images using matplotlib
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))  # Convert to RGB for correct display

# CLAHE Enhanced Image
plt.subplot(1, 2, 2)
plt.title("CLAHE Enhanced Image")
plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))  # Convert to RGB for correct display

plt.show()
