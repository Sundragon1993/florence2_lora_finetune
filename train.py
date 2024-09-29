from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image
import matplotlib.pyplot as plt
from model import Florence2CarDamage
import textwrap

# ckpt_path = '/home/admin/.cache/huggingface/hub/models--microsoft--Florence-2-base/snapshots/ee1f1f163f352801f3b7af6b2b96e4baaa6ff2ff/pytorch_model.bin'
# model = timm.create_model(
#   'davit_base_fl.msft_florence2',
#   pretrained=False,
#   pretrained_cfg_overlay=dict(file=ckpt_path),
# )
# model = model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
torch.cuda.empty_cache()
pretrained_florence = model.eval()
florence_car_model = Florence2CarDamage(pretrained_florence).to(device)
# print(florence_car_model.get_nb_trainable_parameters())

data = load_dataset("tahaman/DamageCarDataset")
# Check the shape of the dataset
train_shape = len(data['train'])
test_shape = len(data['test'])

print(f"Train Dataset Shape: {train_shape} examples")
print(f"Test Dataset Shape: {test_shape} examples")


# Function to run the model on an example
def run_example(task_prompt, text_input, image):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt,
                                                      image_size=(image.width, image.height))
    return parsed_answer


# Test the function with a few examples from your dataset
for idx in range(2):
    image = data['train'][idx]['image']
    description = run_example("Describe the damage to the car.", '', image)
    print(f"Generated Description: {description}")
    plt.imshow(image.resize([350, 350]))
    plt.axis('off')
    plt.show()

from torch.utils.data import Dataset


class DamageCarDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        prompt = "Describe the damage to the car."
        description = example['description']
        image = example['image']
        if image.mode != "RGB":
            image = image.convert("RGB")
        return prompt, description, image


# Create datasets
train_dataset = DamageCarDataset(data['train'])
val_dataset = DamageCarDataset(data['test'])

import os
from torch.utils.data import DataLoader
from tqdm import tqdm


def collate_fn(batch):
    prompts, descriptions, images = zip(*batch)
    inputs = processor(text=list(prompts), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, descriptions


# Create DataLoader
batch_size = 6  # 6
num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers,
                          shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)

# Training Function
from transformers import (AdamW, AutoProcessor, get_scheduler)


def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            inputs, descriptions = batch

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(text=list(descriptions), return_tensors="pt", padding=True,
                                         return_token_type_ids=False).input_ids.to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                inputs, descriptions = batch
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(text=list(descriptions), return_tensors="pt", padding=True,
                                             return_token_type_ids=False).input_ids.to(device)

                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss}")

        # Save model checkpoint
        output_dir = f"./model_checkpoints/epoch_{epoch + 1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir, safe_serialization=False)
        processor.save_pretrained(output_dir)


train_model(train_loader, val_loader, florence_car_model, processor, epochs=10)
