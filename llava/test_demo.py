import os
import math
import torch
from PIL import Image, ImageDraw
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.visual_perception import get_cropped_image


def test_single_sample():
    disable_torch_init()

    model_path = "your_model_path"
    model_base = None
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    sample = {
        "image": "/root/vipuser/images/demo1.png",
        "text": "what is the date of the photo?"
    }

    # 构造对话 prompt
    cur_prompt = sample["text"]
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + (DEFAULT_IMAGE_TOKEN * 2) + DEFAULT_IM_END_TOKEN + '\n' + cur_prompt
    else:
        qs = (DEFAULT_IMAGE_TOKEN * 2) + '\n' + cur_prompt
    conv = conv_templates["llava_v1"].copy()  # 使用 llava_v1 对话模板
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image_path = sample["image"]
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    image = Image.open(image_path).convert('RGB')

    image_tensor = process_images([image], image_processor, model.config)[0].half().cuda()
    print("image_tensor dtype:", image_tensor.dtype)

    general_prompt = 'Write a general description of the image.'
    cropped_image_tensor, bbox = get_cropped_image(image_tensor, cur_prompt, general_prompt, model,
                                         image_processor, tokenizer,image, method="rel_att")
    cropped_image_tensor = cropped_image_tensor.half().cuda()
    print("cropped_image_tensor dtype:", cropped_image_tensor.dtype)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[image_tensor, cropped_image_tensor],
            image_sizes=[image.size, (bbox[2] - bbox[0], bbox[3] - bbox[1])],
            do_sample=True if 0.2 > 0 else False,
            temperature=0.2,
            top_p=0.9,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print("模型推理结果：")
    print(outputs)


if __name__ == '__main__':
    test_single_sample()
