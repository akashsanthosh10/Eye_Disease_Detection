import cv2
import os
import shutil
import numpy as np

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
    #image = cv2.resize(image, (224, 224))
    #image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image


# Function to process a folder of images
def process_images_in_folder(input_folder, output_folder):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each category folder inside the input folder (e.g., cataract, glaucoma, etc.)
    for category in os.listdir(input_folder):
        category_path = os.path.join(input_folder, category)
        
        # Create corresponding category folder in the output folder
        category_output_folder = os.path.join(output_folder, category)
        if not os.path.exists(category_output_folder):
            os.makedirs(category_output_folder)
        
        # Process each image in the category
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            # Apply CLAHE
            enhanced_image = preprocess_image(image_path)
            enhanced_image=cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
            # Save the enhanced image to the new folder
            output_image_path = os.path.join(category_output_folder, image_name)
            cv2.imwrite(output_image_path, enhanced_image)

            print(f"Processed and saved: {output_image_path}")

# Example usage
input_folder = 'dataset'  # The root folder of your dataset
output_folder = 'enhnaced_dataset_3'  # Folder where enhanced images will be saved

# Process all images in the dataset and apply CLAHE
process_images_in_folder(input_folder, output_folder)
