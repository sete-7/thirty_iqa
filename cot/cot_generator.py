import os
import json
import numpy as np
import openai
from typing import List, Dict
from tqdm import tqdm


# ========================================================================
# Structured CoT prompt — forces <think> reasoning + <bbox> defect output
# ========================================================================

SYSTEM_PROMPT = """\
你是一位专业的图像质量评估专家。多位专家模型对该图片的打分产生了较大分歧。\
你需要综合图片内容、生成提示词和各专家分数，进行深入分析。

【输出格式要求——严格遵守】
1. 先输出你的思考过程，使用 <think> ... </think> 标签包裹。
2. 在思考过程中，如果发现图片存在局部缺陷（如模糊、伪影、畸变、不一致等），\
   必须使用 <bbox> [x1, y1, x2, y2] </bbox> 标注缺陷区域的归一化坐标 (0~1)。\
   可以标注多个缺陷区域。
3. 最后给出 200 字以内的分析总结和一个最终综合评价分数 (0-100)，\
   使用 <summary> ... </summary> 和 <final_score> ... </final_score> 标签包裹。

输出示例：
<think>
该图片整体构图较好，但右下角存在明显模糊 <bbox> [0.65, 0.70, 0.95, 0.98] </bbox>，\
左上角天空区域出现色彩伪影 <bbox> [0.02, 0.01, 0.30, 0.25] </bbox>。\
语义与提示词基本一致，但空间布局略有偏差……
</think>
<summary>
图片整体质量中等偏上，语义匹配度高，但存在局部模糊和色彩伪影，空间布局有小幅偏差。
</summary>
<final_score>62</final_score>
"""


def build_user_prompt(item: Dict) -> str:
    """Build the user-side prompt with all expert scores."""
    prompt_text = item.get("prompt", "无提示词")

    lines = [
        f"生成提示词: {prompt_text}",
        "",
        "=== 专家打分参考 ===",
        f"  UniPercept (感知质量)  : {item.get('unipercept_score', 'N/A')}",
        f"  Q-Insight+/Q-Probe (局部缺陷评分): {item.get('q_insight_score', 'N/A')}",
        f"  HPSv3 (语义匹配)       : {item.get('hpsv3_score', 'N/A')}",
        f"  SpatialScore (空间布局) : {item.get('spatial_score', 'N/A')}",
    ]

    # Include Q-Insight+ detected regions if available
    regions = item.get("q_insight_regions", [])
    if regions:
        lines.append("")
        lines.append("=== Q-Insight+ 扫视发现的局部缺陷与得分 ===")
        for r in regions:
            lines.append(
                f"  {r.get('label', 'defect')} "
                f"bbox={r.get('bbox', [])} "
                f"local_score={r.get('local_score', 0):.2f}"
            )

    lines.append("")
    lines.append(
        "请按照系统提示词中的格式要求，输出 <think>, <bbox>, <summary>, <final_score>。"
    )
    return "\n".join(lines)


def generate_cot_for_image(client: openai.OpenAI, model_name: str, item: Dict) -> Dict:
    """Call LLM API to generate structured CoT with <think> and <bbox>."""
    user_prompt = build_user_prompt(item)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=800,
            temperature=0.7,
        )
        cot_text = response.choices[0].message.content
    except Exception as e:
        print(f"[CoT] LLM API error: {e}")
        cot_text = "<think>API 调用失败</think><summary>无法生成分析</summary><final_score>0</final_score>"

    return {
        "image_id": item.get("image_id", item.get("image_path")),
        "cot_analysis": cot_text,
    }


def compute_variance(item: Dict) -> float:
    """
    Compute score variance across all four expert dimensions.
    Scores are first min-max normalized to [0,1] conceptually (here we just use raw variance
    because normalization requires global context; for per-item filtering raw variance suffices).
    """
    scores = []
    for key in ("unipercept_score", "q_insight_score", "hpsv3_score", "spatial_score"):
        val = item.get(key)
        if val is not None:
            scores.append(float(val))
    if len(scores) < 2:
        return 0.0
    return float(np.var(scores))


def process_cot_generation(
    scores_jsonl: str = "expert_scores.jsonl",
    output_jsonl: str = "train_reasoning_data.jsonl",
    variance_threshold: float = 1.0,
):
    """
    Reads expert scores, finds controversial images (high variance),
    generates structured CoT with checkpointing every 100 items.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("LLM_MODEL_NAME", "qwen-vl-max")

    if not api_key:
        print("WARNING: OPENAI_API_KEY not set. LLM calls will fail.")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # 1. Load already-processed IDs
    processed_ids = set()
    if os.path.exists(output_jsonl):
        print(f"Found existing progress in {output_jsonl}. Loading...")
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    j = json.loads(line)
                    processed_ids.add(j.get("image_id", j.get("image_path")))

    # 2. Read scored items and filter controversial ones
    if not os.path.exists(scores_jsonl):
        raise FileNotFoundError(f"Expert scores not found: {scores_jsonl}")

    controversial = []
    with open(scores_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            img_id = item.get("image_id", item.get("image_path"))
            if img_id in processed_ids:
                continue
            var = compute_variance(item)
            item["score_variance"] = var
            if var > variance_threshold:
                controversial.append(item)

    if not controversial:
        print("No remaining controversial items to process.")
        return

    print(f"Found {len(controversial)} controversial items (var > {variance_threshold})")

    # 3. Generate CoT with checkpointing
    buffer = []
    with open(output_jsonl, "a", encoding="utf-8") as f:
        for idx, item in enumerate(tqdm(controversial, desc="Generating CoT")):
            cot_info = generate_cot_for_image(client, model_name, item)
            final_item = {**item, **cot_info}
            buffer.append(json.dumps(final_item, ensure_ascii=False))

            if (idx + 1) % 100 == 0:
                f.write("\n".join(buffer) + "\n")
                f.flush()
                buffer = []
                print(f"  Checkpoint saved at {idx + 1} CoT items.")

        if buffer:
            f.write("\n".join(buffer) + "\n")
            f.flush()
            print("  Final CoT checkpoint saved.")


if __name__ == "__main__":
    pass
