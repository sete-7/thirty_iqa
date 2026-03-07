import os
import json
import numpy as np
import openai
from typing import List, Dict
from tqdm import tqdm

def generate_cot_for_image(client: openai.OpenAI, model_name: str, item: Dict) -> Dict:
    """
    Calls LLM API to generate a CoT evaluation.
    """
    prompt_text = item.get("prompt", "No prompt provided.")
    system_prompt = (
        "三位专家对该图片的语义、质量和美学打分分歧较大。"
        "请结合图片和生成的提示词，给出200字以内的分析理由，并给出最终综合评价分数。"
    )
    user_prompt = (
        f"提示词: {prompt_text}\n"
        f"专家打分参考: 语义->{item.get('semantic_score', 0):.2f}, "
        f"质量->{item.get('quality_score', 0):.2f}, 美学->{item.get('aesthetic_score', 0):.2f}\n"
        "请输出你的分析小作文和综合分数。"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        cot_text = response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        cot_text = "API Call Failed."

    return {
        "image_id": item.get("image_id", item.get("image_path")),
        "cot_analysis": cot_text
    }

def process_cot_generation(scores_jsonl: str, output_jsonl: str = "train_reasoning_data.jsonl", variance_threshold: float = 1.0):
    """
    Reads scores, calculates variance, generating CoT for controversial ones with checkpointing.
    Saves state every 100 items.
    """
    # Initialize OpenAI client 
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("LLM_MODEL_NAME", "qwen-vl-max")

    if not api_key:
        print("WARNING: OPENAI_API_KEY environment variable is not set.")
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # 1. Read existing progress
    processed_ids = set()
    if os.path.exists(output_jsonl):
        print(f"Found existing progress in {output_jsonl}. Loading...")
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    j = json.loads(line)
                    processed_ids.add(j.get("image_id", j.get("image_path")))
                    
    # 2. Read scored items to find controversial ones
    if not os.path.exists(scores_jsonl):
        raise FileNotFoundError(f"Expert scores not found at {scores_jsonl}")

    controversial_data = []
    with open(scores_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            img_id = item.get("image_id", item.get("image_path"))
            if img_id in processed_ids:
                continue

            # Need to normalize scores before variance calculation ideally, 
            # here assuming they are roughly on similar scales or logic handles it
            s1 = item.get("semantic_score", 0.0)
            s2 = item.get("quality_score", 0.0)
            s3 = item.get("aesthetic_score", 0.0)
            
            # Simple variance
            var = np.var([s1, s2, s3])
            
            if var > variance_threshold:
                controversial_data.append(item)

    if not controversial_data:
        print("No remaining controversial items to process.")
        return

    print(f"Found {len(controversial_data)} new controversial items based on var > {variance_threshold}")
    
    buffer = []
    with open(output_jsonl, 'a', encoding='utf-8') as f:
        for idx, item in enumerate(tqdm(controversial_data, desc="Generating CoT")):
            # Update item with CoT
            cot_info = generate_cot_for_image(client, model_name, item)
            
            # Merge original item and cot
            final_item = {**item, **cot_info}
            buffer.append(json.dumps(final_item, ensure_ascii=False))
            
            # Save every 100 items (though CoT might be fewer items total)
            if (idx + 1) % 100 == 0:
                f.write("\n".join(buffer) + "\n")
                f.flush()
                buffer = []
                print(f"  Checkpoint saved at {idx + 1} CoT generated.")
                
        if buffer:
            f.write("\n".join(buffer) + "\n")
            f.flush()
            print("  Final CoT checkpoint saved.")

if __name__ == "__main__":
    pass
