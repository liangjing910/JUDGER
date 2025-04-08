import argparse
import base64
import json
from openai import OpenAI
from helper import sys2


api_key = "your_api_key"
api_base = "https://api.openai.com/v1"
client = OpenAI(api_key=api_key, base_url=api_base)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def gpt_api(sys_prompt, usr_prompt, image_path):

    try:
        full_image_path = f"/datasets/data/gqa/images/{image_path}"
        # Getting the Base64 string
        base64_image = encode_image(full_image_path)

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": usr_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=512,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


def query_why(question_id, question, model_answer, correct_answer, image_path):

    usr = f"""
    ### Question: {question}
    ### Model Answer: {model_answer}
    ### Correct Answer: {correct_answer}
    """

    return gpt_api(sys2, usr, image_path)


def main():
    parser = argparse.ArgumentParser(description='Generate judgment for GQA using GPT-4o')
    parser.add_argument('--input_jsonl', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output_jsonl', type=str, default=None, help='Path to save output JSONL')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--end_idx', type=int, default=-1, help='End index')

    args = parser.parse_args()

    # 读取输入 JSONL 数据
    with open(args.input_jsonl, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    assert start_idx < end_idx, "起始索引必须小于结束索引。"

    output_path = args.output_jsonl if args.output_jsonl else f'judgment_output_img_{start_idx}_{end_idx}.jsonl'

    with open(output_path, 'a', encoding='utf-8') as outfile:
        for idx, item in enumerate(data[start_idx:end_idx], start=start_idx):
            question_id = item['question_id']
            question = item['prompt']
            model_answer = item['text']
            correct_answer = item['answer']
            image_path = item['image']


            if model_answer.lower() != correct_answer.lower():
                judgment = query_why(question_id, question, model_answer, correct_answer, image_path)
                item['judgment'] = judgment
            else:
                item['judgment'] = None

            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')


            if (idx + 1) % 50 == 0:
                outfile.flush()
                print(f"[Checkpoint] Processed {idx + 1} items")

        print(f"完成处理。输出保存至：{output_path}")


if __name__ == '__main__':
    main()
