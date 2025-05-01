from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image


processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("training_output")


label2id = {"cataract": 0, "diabetic_retinopathy": 1, "glaucoma": 2, "normal": 3}
id2label = {0: "cataract", 1: "diabetic_retinopathy", 2: "glaucoma", 3: "normal"}

model.config.label2id = label2id
model.config.id2label = id2label

image_path = "test/cataract/cataract_080.png"  
image = Image.open(image_path)


inputs = processor(image, return_tensors="pt")


#print(inputs['pixel_values'].shape)

model.eval()


with torch.no_grad():
    outputs = model(**inputs)


logits = outputs.logits
predicted_class_id = logits.argmax(-1).item()


predicted_label = model.config.id2label[predicted_class_id]


print(f"Predicted Disease: {predicted_label}")
