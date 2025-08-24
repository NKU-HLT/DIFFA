from .base import Evaluator
import numpy as np
from qa_metrics.pedant import PEDANT


def majority_vote(scores):
    scores = [item.lower() for item in scores]
    final_answer = max(set(scores), key=scores.count)

    # Convert the final answer to True for 'Yes' and False for 'No'
    return True if final_answer == 'yes' else False


class QAEvaluator(Evaluator):
    def __init__(self):
        self.pedant = PEDANT()

    def evaluate(self, data):
        panda_results = [self.pedant.evaluate([item['reference'].lower()], item['response'].lower(), item['prompt'].lower()) for item in data]
        gpt_results = [majority_vote(item['score']) for item in data]
        return {
            'panda': np.mean(panda_results)*100, 'gpt': np.mean(gpt_results) * 100
        }
