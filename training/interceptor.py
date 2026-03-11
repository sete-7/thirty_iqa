import re

class RewardInterceptor:
    """
    Post-processing interceptor for LLM generations (like GRPO reasoning trajectories).
    Ensures that the LLM output strictly adheres to the requested `<think>`, `<bbox>`, and `<score>` formats.
    """
    
    def __init__(self):
        # Regex to match <bbox> [x1, y1, x2, y2] </bbox>
        self.bbox_pattern = re.compile(r"<bbox>\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]\s*</bbox>")
        
        # Regex to extract final score from <score>...</score>
        self.score_pattern = re.compile(r"<score>\s*([0-9.]+)\s*</score>")
        
        # Negative keywords that strongly imply a defect/deduction was found.
        self.negative_keywords = ["模糊", "伪影", "畸变", "缺陷", "不一致", "偏差", "失真", "噪点"]

    def evaluate_output(self, cot_text: str, current_reward_weight: float = 1.0) -> float:
        """
        Evaluates the generated CoT text. 
        Rule: If the output implies a negative evaluation/deduction, but fails to provide
        valid <bbox> coordinates to physically ground the defect, force the reward/weight to 0.
        
        Returns:
            The adjusted reward weight (0.0 if violation, otherwise unchanged).
        """
        # 1. Check if negative evaluation is present
        # Heuristic 1: Score is less than some hypothetical perfection threshold (e.g. < 90)
        # We can extract the score to check
        score_match = self.score_pattern.search(cot_text)
        implied_deduction = False
        
        if score_match:
            try:
                score = float(score_match.group(1))
                if score < 85.0: # Moderate to heavy deduction
                    implied_deduction = True
            except ValueError:
                pass
                
        # Heuristic 2: Explicit negative language in the text
        if any(keyword in cot_text for keyword in self.negative_keywords):
            implied_deduction = True
            
        # 2. Extract bounding boxes
        bboxes = self.bbox_pattern.findall(cot_text)
        has_valid_bboxes = len(bboxes) > 0
        
        # 3. Apply Constraint Logic
        if implied_deduction and not has_valid_bboxes:
            # The model made a deduction but hallucinated/failed to ground it with coordinates.
            print("[Interceptor] Violation detected: Negative evaluation found without corresponding <bbox>. Forcing weight/reward to 0.")
            return 0.0
            
        return current_reward_weight

    def extract_bboxes(self, cot_text: str) -> list:
        """Utility to safely extract parsed bounding boxes as floats."""
        matches = self.bbox_pattern.findall(cot_text)
        parsed = []
        for x1, y1, x2, y2 in matches:
            try:
                parsed.append([float(x1), float(y1), float(x2), float(y2)])
            except ValueError:
                continue
        return parsed

# Example Usage
if __name__ == "__main__":
    interceptor = RewardInterceptor()
    
    # Example 1: Good text
    text1 = "<think>右下角模糊 <bbox>[0.1, 0.2, 0.3, 0.4]</bbox></think><score>70</score>"
    w1 = interceptor.evaluate_output(text1, 1.0)
    print("Test 1 Weight:", w1) # Should be 1.0
    
    # Example 2: Bad text (mentions defect but no bbox)
    text2 = "<think>天空有明显噪点和失真。</think><score>60</score>"
    w2 = interceptor.evaluate_output(text2, 1.0)
    print("Test 2 Weight:", w2) # Should be 0.0
