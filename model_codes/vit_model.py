from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import matplotlib.pyplot as plt
import numpy as np
from evaluate import load


import torch

def collate_fn(examples):
    pixel_values = torch.stack([torch.tensor(example["pixel_values"]) for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}



metric = load("accuracy")
def compute_metrics(eval_pred):

    predictions = np.argmax(eval_pred.predictions, axis=1)
    
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)





dataset = load_dataset("imagefolder", data_dir="dataset")
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224",use_fast=True)
#print(dataset)
#print(processor)

def show_image(image,label):
    
    print("Image size:", image.size)  
    print("Image format:", image.format) 
    print("Label:", label)

   
    plt.imshow(image)
    plt.axis("off") 
    plt.title(f"Label: {label}")
    plt.show()





"""sample = dataset['train'][1]
image = sample['image']
label = sample['label']

inputs = processor(image, return_tensors="pt")
#show_image(image,label)

pixel_values = inputs['pixel_values']
print("Shape of pixel_values:", pixel_values.shape)"""



def preprocess_images(example):
   
    inputs = processor(example['image'], return_tensors="pt")
    example['pixel_values'] = inputs['pixel_values'].squeeze()  
    return example


processed_dataset = dataset.map(preprocess_images)
#print(processed_dataset)

splits = processed_dataset["train"].train_test_split(test_size=0.3)
train_ds = splits['train']
val_ds = splits['test']
#print(train_ds,val_ds)

label2id = {"cataract": 0, "diabetic_retinopathy": 1, "glaucoma": 2, "normal": 3}
id2label = {0: "cataract", 1: "diabetic_retinopathy", 2: "glaucoma", 3: "normal"}

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)



args = TrainingArguments(
  output_dir="training_output",
  per_device_train_batch_size=16,
  eval_strategy="steps",
  num_train_epochs=4,
  fp16=True,
  save_steps=250,
  eval_steps=250,
  logging_steps=100,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)
trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

train_results = trainer.train()
metrics = trainer.evaluate()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


#print(model)