import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img


def preprocess_image(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (224, 224))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image

    return denoised_image
def show_images(original, processed, title_original, title_processed):
    """ Helper function to show images side-by-side """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title(title_original)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed)
    plt.title(title_processed)
    plt.axis('off')

    plt.show()




def process_dataset(dataset_path):
    """
    Process all images in the given dataset directory.
    :param dataset_path: str, path to the dataset directory
    """
    # List all image files in the directory
    for filename in os.listdir(dataset_path):
        # Check if the file is an image (e.g., .png, .jpg)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(dataset_path, filename)
            print(f"Processing {filename}...")

            # Process the image
            processed_image = preprocess_image(image_path)

            # Load the original image for comparison
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            # Display the original and processed images side-by-side
            show_images(original_image, processed_image, "Original Image", "Processed Image")

# Example usage
dataset_path = 'dataset/normal'  # Replace with the path to your dataset folder
#process_dataset(dataset_path)


processed_image = preprocess_image("dataset\diabetic_retinopathy\\10947_left.jpeg")
original_image = cv2.imread("dataset\diabetic_retinopathy\\10947_left.jpeg")
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
show_images(original_image, processed_image, "Original Image", "Processed Image")

















"""

def show_images(original, processed):
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title("Processed Image")
    plt.axis('off')

    plt.show()

# Example usage:
image_path = 'dataset\glaucoma\Glaucoma_097.png'  # Replace with your image path
processed_image = preprocess_image(image_path)

# Display the original vs processed image
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # For comparison
show_images(original_image, processed_image)

"""