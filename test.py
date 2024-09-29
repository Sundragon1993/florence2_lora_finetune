from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
import torch
from PIL import Image
import matplotlib.pyplot as plt
import textwrap
from model import Florence2CarDamage
from safetensors.torch import load_file


class FlorenceCarDamageDescriber:
    def __init__(self, model_path, state_dict_path, processor_path, device=None):
        # Set up the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        # Load model config, processor, and architecture
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

        # Wrap the model with custom Florence2CarDamage class and load the saved state dict
        self.model = Florence2CarDamage(model).to(self.device)
        state_dict = torch.load(state_dict_path)
        self.model.backbone.load_state_dict(state_dict)
        self.model = self.model.eval()

    def describe_damage(self, task_prompt, text_input, image_path):
        # Construct the full prompt
        prompt = task_prompt + (text_input if text_input else '')

        # Load and preprocess the image
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Tokenize inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        # Generate output
        generated_ids = self.model.backbone.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt,
                                                               image_size=(image.width, image.height))

        # Ensure parsed_answer is a string
        if isinstance(parsed_answer, dict):
            parsed_answer = str(parsed_answer)

        plt.imshow(image)
        plt.axis('off')
        plt.show()

        wrapped_description = textwrap.fill(parsed_answer, width=120)
        print(f"Generated Description:\n{wrapped_description}")
        return parsed_answer


model_path = "/home/admin/florence2/model_checkpoints/epoch_10"
state_dict_path = '/home/admin/florence2/model_checkpoints/epoch_10/pytorch_model.bin'
processor_path = "/home/admin/florence2/model_checkpoints/epoch_10"
image_path = "/home/admin/florence2/test_img/damage-1200x800.jpeg"

car_damage_describer = FlorenceCarDamageDescriber(model_path, state_dict_path, processor_path)
description = car_damage_describer.describe_damage("Describe the damage to the car.", '', image_path)
