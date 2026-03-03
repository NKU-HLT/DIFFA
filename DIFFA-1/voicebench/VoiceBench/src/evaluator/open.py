import re
from .base import Evaluator
import numpy as np


def extract_rating(llm_output):
    """
    Extracts the rating in the format [[number]] from the LLM output.

    Args:
    - llm_output (str): The response from the LLM containing the evaluation and rating.

    Returns:
    - int: The extracted rating, or None if the rating is not found.
    """
    # Define the regular expression pattern to match the rating in the format [[number]]
    pattern = r"\[\[(\d+)\]\]"

    # Search for the pattern in the LLM output
    match = re.search(pattern, llm_output)

    if match:
        # Convert the matched rating to an integer and return it
        return int(match.group(1))
    else:
        # Return None if no rating is found
        # return None
        raise NotImplementedError


class OpenEvaluator(Evaluator):
    def evaluate(self, data):
        scores = []
        for item in data:
            for score in item['score']:
                try:
                    score = float(score)
                except Exception as e:
                    score = extract_rating(score)
                scores.append(score)
        return {'gpt': np.mean(scores)}

