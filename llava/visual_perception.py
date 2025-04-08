import torch

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token_uns
from utils.utils import bbox_from_att_image_adaptive


def get_cropped_image(image_tensor, prompt, general_prompt, model, processor, tokenizer, image, method="rel_att"):

    if method == "rel_att":
        with torch.no_grad():
            att_map = rel_attention_llava(image_tensor, prompt, general_prompt, model, tokenizer)
    else:
        raise ValueError("Unknown ViCrop method")

    desired_size = 336
    bbox = bbox_from_att_image_adaptive(att_map, image.size, desired_size)
    crop_image = image.crop(bbox)
    crop_image_tensor = processor.preprocess(crop_image, return_tensors='pt')['pixel_values'][0]
    return crop_image_tensor, bbox



NUM_IMG_TOKENS = 576
NUM_PATCHES = 24
PATCH_SIZE = 14
IMAGE_RESOLUTION = 336
ATT_LAYER = 14


def rel_attention_llava(
    image_tensor: torch.Tensor,
    prompt: str,
    general_prompt: str,
    model,
    tokenizer
):
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # [1,3,H,W]

    combined_prompt = f"<image>\n{prompt}"

    inputs_s = tokenizer_image_token_uns(
        combined_prompt,
        tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    )

    forward_inputs_s = {
        "input_ids": inputs_s["input_ids"].to(model.device),
        "attention_mask": inputs_s["attention_mask"].to(model.device),
        # "images": image_tensor.to(model.device, dtype=torch.bfloat16),
        "images": image_tensor.to(model.device, dtype=torch.half),
        "labels": inputs_s["input_ids"].clone().to(model.device)
    }

    token_ids_s = forward_inputs_s["input_ids"][0].tolist()
    pos_s = token_ids_s.index(IMAGE_TOKEN_INDEX)

    outputs_s = model(**forward_inputs_s, output_attentions=True, return_dict=True)
    # att_map_s = outputs_s[0].attentions[ATT_LAYER][0, :, -1, pos_s:pos_s + NUM_IMG_TOKENS].mean(dim=0)
    att_map_s = outputs_s["attentions"][ATT_LAYER][0, :, -1, pos_s:pos_s + NUM_IMG_TOKENS].mean(dim=0)

    att_map_s = att_map_s.float().detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)

    combined_general_prompt = f"<image>\n{general_prompt}"

    inputs_g = tokenizer_image_token_uns(
        combined_general_prompt,
        tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    )


    forward_inputs_g = {
        "input_ids": inputs_g["input_ids"].to(model.device),
        "attention_mask": inputs_g["attention_mask"].to(model.device),
        # "images": image_tensor.to(model.device, dtype=torch.bfloat16),
        "images": image_tensor.to(model.device, dtype=torch.half),
        "labels": inputs_g["input_ids"].clone().to(model.device)
    }

    token_ids_g = forward_inputs_g["input_ids"][0].tolist()
    pos_g = token_ids_g.index(IMAGE_TOKEN_INDEX)

    outputs_g = model(**forward_inputs_g, output_attentions=True, return_dict=True)
    # att_map_g = outputs_g[0].attentions[ATT_LAYER][0, :, -1, pos_g:pos_g + NUM_IMG_TOKENS].mean(dim=0)
    att_map_g = outputs_g["attentions"][ATT_LAYER][0, :, -1, pos_s:pos_s + NUM_IMG_TOKENS].mean(dim=0)
    att_map_g = att_map_g.float().detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)

    final_att_map = att_map_s / att_map_g

    return final_att_map