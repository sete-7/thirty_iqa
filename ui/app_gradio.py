import gradio as gr
import os
import csv
import json
import random

# For demonstration, assume we have a list of image pairs
# In practice you'd generate this from your dataset
DATA_POOL = []

def load_data_pool(pairs_jsonl: str = "app_pairs.jsonl"):
    global DATA_POOL
    DATA_POOL = []
    if os.path.exists(pairs_jsonl):
        with open(pairs_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    DATA_POOL.append(json.loads(line))
    # Dummy mock data if none exists
    if not DATA_POOL:
        DATA_POOL.append({
            "prompt": "Test Prompt 1: A futuristic city",
            "img_left": "test1_a.jpg", 
            "img_right": "test1_b.jpg"
        })
        DATA_POOL.append({
            "prompt": "Test Prompt 2: An ancient wizard",
            "img_left": "test2_a.jpg", 
            "img_right": "test2_b.jpg"
        })

STATE = {"current_pair_idx": 0}

def get_next_pair():
    # Loop back if we reach the end
    idx = STATE["current_pair_idx"] % max(1, len(DATA_POOL))
    
    if not DATA_POOL:
        return "No data", None, None
        
    pair = DATA_POOL[idx]
    
    # We could optionally shuffle left/right here for true blindness
    # (just need to track the actual paths)
    img1, img2 = pair["img_left"], pair["img_right"]
    STATE["actual_left"] = img1
    STATE["actual_right"] = img2
    
    return pair["prompt"], img1, img2

def record_vote(choice: str):
    """
    choice is either "left" or "right"
    """
    csv_file = "human_votes.csv"
    file_exists = os.path.isfile(csv_file)
    
    if STATE.get("actual_left") and STATE.get("actual_right"):
        win_path = STATE["actual_left"] if choice == "left" else STATE["actual_right"]
        lose_path = STATE["actual_right"] if choice == "left" else STATE["actual_left"]
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["win_path", "lose_path"])
            writer.writerow([win_path, lose_path])
            
    # Advance to next pair
    STATE["current_pair_idx"] += 1
    return get_next_pair()

def vote_left():
    return record_vote("left")

def vote_right():
    return record_vote("right")

def build_ui():
    load_data_pool()
    
    with gr.Blocks(title="Blind Test Image Preference") as demo:
        gr.Markdown("<h1 style='text-align: center;'>Image Preference Blind Test</h1>")
        gr.Markdown("<h3 style='text-align: center;'>Select which image better matches the prompt or looks higher quality!</h3>")
        
        prompt_text = gr.Markdown(value="Loading...", elem_classes="prompt-box")
        
        with gr.Row():
            img_left = gr.Image(label="Image A", interactive=False)
            img_right = gr.Image(label="Image B", interactive=False)
            
        with gr.Row():
            btn_left = gr.Button("👈 左边更好 (Left is Better)", variant="primary")
            btn_right = gr.Button("右边更好 👉 (Right is Better)", variant="primary")

        # Initialize first
        demo.load(fn=get_next_pair, outputs=[prompt_text, img_left, img_right])
        
        # Button actions
        btn_left.click(fn=vote_left, outputs=[prompt_text, img_left, img_right])
        btn_right.click(fn=vote_right, outputs=[prompt_text, img_left, img_right])
        
    return demo

if __name__ == "__main__":
    # Create mock dummy images for testing so it doesn't crash on start
    from PIL import Image
    for f in ["test1_a.jpg", "test1_b.jpg", "test2_a.jpg", "test2_b.jpg"]:
        if not os.path.exists(f):
            Image.new('RGB', (256, 256), color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))).save(f)

    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
