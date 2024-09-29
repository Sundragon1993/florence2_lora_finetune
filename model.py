import timm
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import matplotlib.pyplot as plt
import textwrap
from lora_utils import QkvWithLoRA, KvWithLoRA, LinearWithLoRA
from peft import LoraConfig, get_peft_model


class Florence2CarDamage(torch.nn.Module):
    def __init__(self, pretrained_model, lora_type='full', lora_rank=32, lora_alpha=32, do_k_lora=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = self.init_florence2_lora(pretrained_model, lora_type, lora_rank, lora_alpha,
                                                 do_k_lora)

    def init_florence2_lora(self, pretrained_model, lora_type='full', lora_rank=32, lora_alpha=32,
                            do_k_lora=True):
        model = pretrained_model
        torch.cuda.empty_cache()
        model = model.eval()
        if lora_type == 'qv':
            do_k_lora = False
            for name, params in model.named_parameters():
                # print(f'name, params: {name} {params}')
                if 'qkv.weight' in name:
                    index = int(name.split('.')[2])
                    jndex = int(name.split('.')[3])
                    # print(f'index: {index} {jndex}')
                    # print(
                    #     f'\n qkv.weight name: {name}')
                    if 'spatial_block' in name:
                        model.vision_tower.blocks[index][jndex].spatial_block.window_attn.fn.qkv = QkvWithLoRA(
                            model.vision_tower.blocks[index][jndex].spatial_block.window_attn.fn.qkv, lora_rank,
                            lora_alpha,
                            do_k_lora)
                    elif 'channel_block' in name:
                        model.vision_tower.blocks[index][jndex].channel_block.channel_attn.fn.qkv = QkvWithLoRA(
                            model.vision_tower.blocks[index][jndex].channel_block.channel_attn.fn.qkv, lora_rank,
                            lora_alpha,
                            do_k_lora)
        elif lora_type == 'full':
            for name, params in model.named_parameters():
                if 'qkv.weight' in name:
                    index = int(name.split('.')[2])
                    jndex = int(name.split('.')[3])
                    # print(f'index: {index} {jndex}')
                    # print(
                    #     f'\n qkv.weight name: {name}')  # vision_tower.blocks.2.5.channel_block.channel_attn.fn.qkv.weight
                    if 'spatial_block' in name:
                        model.vision_tower.blocks[index][jndex].spatial_block.window_attn.fn.qkv = QkvWithLoRA(
                            model.vision_tower.blocks[index][jndex].spatial_block.window_attn.fn.qkv, lora_rank,
                            lora_alpha,
                            do_k_lora)
                    elif 'channel_block' in name:
                        model.vision_tower.blocks[index][jndex].channel_block.channel_attn.fn.qkv = QkvWithLoRA(
                            model.vision_tower.blocks[index][jndex].channel_block.channel_attn.fn.qkv, lora_rank,
                            lora_alpha,
                            do_k_lora)

                elif 'fn.proj.weight' in name and 'convs' not in name:
                    # print(f'\n [D] Proj weight: {name}')
                    index = int(name.split('.')[2])
                    jndex = int(name.split('.')[3])
                    # print(f'index: {index} {jndex}')
                    if 'channel_block' in name:
                        model.vision_tower.blocks[index][jndex].channel_block.channel_attn.fn.proj = LinearWithLoRA(
                            model.vision_tower.blocks[index][jndex].channel_block.channel_attn.fn.proj, lora_rank,
                            lora_alpha)
                    elif 'spatial_block' in name:
                        model.vision_tower.blocks[index][jndex].spatial_block.window_attn.fn.proj = LinearWithLoRA(
                            model.vision_tower.blocks[index][jndex].spatial_block.window_attn.fn.proj, lora_rank,
                            lora_alpha)
                    # model.blocks[index].attn.proj = \
                    #     LinearWithLoRA(model.blocks[index].attn.proj, lora_rank, lora_alpha)
                elif 'net.fc1.weight' in name:
                    index = int(name.split('.')[2])
                    jndex = int(name.split('.')[3])
                    # print(f'index: {index} {jndex}')
                    if 'channel_block' in name:
                        model.vision_tower.blocks[index][jndex].channel_block.ffn.fn.net.fc1 = LinearWithLoRA(
                            model.vision_tower.blocks[index][jndex].channel_block.ffn.fn.net.fc1, lora_rank, lora_alpha)
                    elif 'spatial_block' in name:
                        model.vision_tower.blocks[index][jndex].spatial_block.ffn.fn.net.fc1 = LinearWithLoRA(
                            model.vision_tower.blocks[index][jndex].spatial_block.ffn.fn.net.fc1, lora_rank, lora_alpha)
                    # model.blocks[index].mlp.fc1 = \
                    #     LinearWithLoRA(model.blocks[index].mlp.fc1, lora_rank, lora_alpha)
                elif 'net.fc2.weight' in name:
                    index = int(name.split('.')[2])
                    jndex = int(name.split('.')[3])
                    # print(f'\n [D] fc2 weight: {name}')
                    if 'channel_block' in name:
                        model.vision_tower.blocks[index][jndex].channel_block.ffn.fn.net.fc2 = LinearWithLoRA(
                            model.vision_tower.blocks[index][jndex].channel_block.ffn.fn.net.fc2, lora_rank, lora_alpha)
                    elif 'spatial_block' in name:
                        model.vision_tower.blocks[index][jndex].spatial_block.ffn.fn.net.fc2 = LinearWithLoRA(
                            model.vision_tower.blocks[index][jndex].spatial_block.ffn.fn.net.fc2, lora_rank, lora_alpha)
                elif 'k_proj.weight' in name:
                    # print(f'\n [D] k_proj weight: {name}')
                    index = int(name.split('.')[4])
                    if 'model.encoder' in name:
                        model.language_model.model.encoder.layers[index].self_attn.k_proj = LinearWithLoRA(
                            model.language_model.model.encoder.layers[index].self_attn.k_proj, lora_rank, lora_alpha)
                    elif 'model.decoder' in name:
                        if 'encoder_attn' in name:
                            model.language_model.model.decoder.layers[index].encoder_attn.k_proj = LinearWithLoRA(
                                model.language_model.model.decoder.layers[index].encoder_attn.k_proj, lora_rank,
                                lora_alpha)
                        elif 'self_attn' in name:
                            model.language_model.model.decoder.layers[index].self_attn.k_proj = LinearWithLoRA(
                                model.language_model.model.decoder.layers[index].self_attn.k_proj, lora_rank,
                                lora_alpha)
                elif 'v_proj.weight' in name:
                    # print(f'\n [D] v_proj weight: {name}')
                    index = int(name.split('.')[4])
                    if 'model.encoder' in name:
                        model.language_model.model.encoder.layers[index].self_attn.v_proj = LinearWithLoRA(
                            model.language_model.model.encoder.layers[index].self_attn.v_proj, lora_rank, lora_alpha)
                    elif 'model.decoder' in name:
                        if 'encoder_attn' in name:
                            model.language_model.model.decoder.layers[index].encoder_attn.v_proj = LinearWithLoRA(
                                model.language_model.model.decoder.layers[index].encoder_attn.v_proj, lora_rank,
                                lora_alpha)
                        elif 'self_attn' in name:
                            model.language_model.model.decoder.layers[index].self_attn.v_proj = LinearWithLoRA(
                                model.language_model.model.decoder.layers[index].self_attn.v_proj, lora_rank,
                                lora_alpha)
                elif 'out_proj.weight' in name:
                    # print(f'\n [D] out_proj weight: {name}')
                    index = int(name.split('.')[4])
                    if 'model.encoder' in name:
                        model.language_model.model.encoder.layers[index].self_attn.out_proj = LinearWithLoRA(
                            model.language_model.model.encoder.layers[index].self_attn.out_proj, lora_rank, lora_alpha)
                    elif 'model.decoder' in name:
                        if 'encoder_attn' in name:
                            model.language_model.model.decoder.layers[index].encoder_attn.out_proj = LinearWithLoRA(
                                model.language_model.model.decoder.layers[index].encoder_attn.out_proj, lora_rank,
                                lora_alpha)
                        elif 'self_attn' in name:
                            model.language_model.model.decoder.layers[index].self_attn.out_proj = LinearWithLoRA(
                                model.language_model.model.decoder.layers[index].self_attn.out_proj, lora_rank,
                                lora_alpha)
                elif 'v_proj.weight' in name:
                    # print(f'\n [D] v_proj weight: {name}')
                    index = int(name.split('.')[4])
                    if 'model.encoder' in name:
                        model.language_model.model.encoder.layers[index].self_attn.v_proj = LinearWithLoRA(
                            model.language_model.model.encoder.layers[index].self_attn.v_proj, lora_rank, lora_alpha)
                    elif 'model.decoder' in name:
                        if 'encoder_attn' in name:
                            model.language_model.model.decoder.layers[index].encoder_attn.v_proj = LinearWithLoRA(
                                model.language_model.model.decoder.layers[index].encoder_attn.v_proj, lora_rank,
                                lora_alpha)
                        if 'self_attn' in name:
                            model.language_model.model.decoder.layers[index].self_attn.v_proj = LinearWithLoRA(
                                model.language_model.model.decoder.layers[index].self_attn.v_proj, lora_rank,
                                lora_alpha)
        return model

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.backbone.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif not hasattr(param, "quant_storage"):
                    num_bytes = 1
                else:
                    num_bytes = param.quant_storage.itemsize
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
    def forward(self,*args, **kwargs):
        return self.backbone(*args, **kwargs)
    def save_pretrained(self,*args, **kwargs):
        return self.backbone.save_pretrained(*args, **kwargs)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_florence = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(
    device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
torch.cuda.empty_cache()

'''
Timm Davit
'''
# ckpt_path = '/home/admin/.cache/huggingface/hub/models--microsoft--Florence-2-base/snapshots/ee1f1f163f352801f3b7af6b2b96e4baaa6ff2ff/pytorch_model.bin'
# pretrained_florence = timm.create_model(
#   'davit_base_fl.msft_florence2',
#   pretrained=False,
#   pretrained_cfg_overlay=dict(file=ckpt_path),
# )

pretrained_florence = pretrained_florence.eval()
florence_car_model = Florence2CarDamage(pretrained_florence)
print(florence_car_model.get_nb_trainable_parameters())


'''
LoRA Peft
'''
# from peft import LoraConfig, get_peft_model
# CHECKPOINT = "microsoft/Florence-2-base-ft"
# REVISION = 'refs/pr/6'
# TARGET_MODULES = [
#     "q_proj", "out_proj", "k_proj", "v_proj", "qkv", "fc1",
#     "linear", "Conv2d", "lm_head", "fc2", "fn.proj"
# ]
#
# config = LoraConfig(
#     r=32,
#     lora_alpha=32,
#     target_modules=TARGET_MODULES,
#     task_type="CAUSAL_LM",
#     lora_dropout=0.05,
#     bias="none",
#     inference_mode=False,
#     use_rslora=True,
#     init_lora_weights="gaussian",
#     revision=REVISION
# )
#
# peft_model = get_peft_model(pretrained_florence, config)
# peft_model.print_trainable_parameters()
