import re
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-formatted text."""
    match = re.search('<answer>(.*)</answer>', text, re.DOTALL)
    if match:
        answer = match.group(1)
    else:
        answer = ''
    return answer.strip()

def extract_hash_answer(text: str) -> Optional[str]:
    """Extract answer from GSM8K format (after ####)."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_dataset(split="train"):
    """
    Get GSM8K dataset
    Args:
        split: train or test
    Returns:
        dataset
    """
    data = load_dataset("openai/gsm8k", name="main", split=split)
    return data

def correctness_reward_func(prompts: List[Dict], 
                          completions: List[Dict], 
                          answer: List[str], 
                          **kwargs) -> List[float]:
    """
    Calculate correctness reward based on answer matching.
    """
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20, 
          f"Question:\n{q}", 
          f"\nResponse:\n{responses[0]}", 
          f"\nExtracted:\n{extracted_responses[0]}", 
          f"\nAnswer:\n{answer[0]}")
    return [1 if a in r else 0.0 for r, a in zip(extracted_responses, answer)]

def soft_format_reward_func(completions: List[Dict], **kwargs) -> List[float]:
    """
    Calculate reward for responses that loosely follow the XML format.
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [2 if match else 0.0 for match in matches]

def strict_format_reward_func(completions: List[Dict], **kwargs) -> List[float]:
    """
    Calculate reward for responses that strictly follow the XML format.
    """
    pattern = r"^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [4 if match else 0.0 for match in matches] 