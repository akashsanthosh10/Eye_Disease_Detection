from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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
    #show_image(image,"input")
    image = crop_image_from_gray(image)
    #image = cv2.resize(image, (224, 224))
    #image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image

def show_image(image,label):
    

   
    plt.imshow(image)
    plt.axis("off") 
    plt.title(f"Label: {label}")
    plt.show()


def predict(image_path):
    image = preprocess_image(image_path)
    #show_image(image,"input")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)


    logits = outputs.logits
    predicted_class_id = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_id]
    probabilities = torch.softmax(logits, dim=-1)
    predicted_probability = probabilities[0, predicted_class_id].item()
    #print(logits,predicted_probability)
    return predicted_label,predicted_probability

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained("training_result\\resnet_version6")

print(processor)
label2id = {"cataract": 0, "diabetic_retinopathy": 1, "glaucoma": 2, "normal": 3}
id2label = {0: "cataract", 1: "diabetic_retinopathy", 2: "glaucoma", 3: "normal"}
model.eval()
model.config.label2id = label2id
model.config.id2label = id2label

#image_path = "test_image\diabetic_retinopathy\\37332_left.jpeg"  
#predicted_label,predicted_probability = predict(image_path)
#print(f"Predicted Label: {predicted_label}({predicted_probability})")


image_folder = r"test_image\\glaucoma"  # Change this to your image folder path
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]
correct=0
incorrect=0
# Iterate through the images and predict
for image_path in image_paths:
    predicted_label,predicted_probability = predict(image_path)
    true_label=image_folder.split('\\')[2]
    if predicted_label==true_label:
        correct=correct+1
    else:
        incorrect=incorrect+1
    print(f"Predicted Label for {image_path}: {predicted_label}({predicted_probability})")

total_images = correct + incorrect
accuracy = correct / total_images if total_images > 0 else 0

print(f"Total Images: {total_images}")
print(f"Correct Predictions: {correct}")
print(f"Incorrect Predictions: {incorrect}")
print(f"Accuracy: {accuracy * 100:.2f}%")







