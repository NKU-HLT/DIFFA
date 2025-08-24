import json
import argparse
from collections import defaultdict

def load_jsonl_data(jsonl_path):
    """Load data from the provided JSONL file."""
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {line.strip()}")
                print(e)
    return records

def calculate_accuracy(data):
    """
    Calculate accuracy for:
    1. (category, sub-category)
    2. (category, sub-category, sub-sub-category)
    """
    # 二级统计（task + sub-category）
    level2_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    
    # 三级统计（task + sub-category + sub-sub-category）
    level3_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0})))

    task_avg_stats = defaultdict(lambda: {"total_correct": 0, "total_count": 0})

    total_correct = 0
    total_count = 0

    for record in data:
        task = record.get('category', '')
        subcat = record.get('sub-category', '')
        subsubcat = record.get('sub-sub-category', '')

        response = record.get('response', '')
        try:
            predict = response.strip().replace('\n', '')
        except:
            predict = ''

        model_predict = None
        if predict and predict != 'None':
            if predict[0] in 'ABCD':
                model_predict = predict[0]
            elif len(predict) > 1 and predict[-2] in 'ABCD':
                model_predict = predict[-2]
            else:
                print(f'Wrong format response: {predict}')
        else:
            print(f'Empty or None response.')

        # Get correct answer and choices
        answer_gt = record.get('answer_gt', '')
        choices = {
            'A': record.get('choice_a', ''),
            'B': record.get('choice_b', ''),
            'C': record.get('choice_c', ''),
            'D': record.get('choice_d', '')
        }

        # Update total count regardless of format issues
        level2_stats[task][subcat]["total"] += 1
        level3_stats[task][subcat][subsubcat]["total"] += 1
        total_count += 1

        # Update correct count only if prediction is valid
        if model_predict and choices.get(model_predict) == answer_gt:
            level2_stats[task][subcat]["correct"] += 1
            level3_stats[task][subcat][subsubcat]["correct"] += 1
            total_correct += 1

    # 计算每个task的平均精度
    for task, subcats in level2_stats.items():
        task_correct = 0
        task_total = 0
        for subcat, stats in subcats.items():
            task_correct += stats["correct"]
            task_total += stats["total"]
        task_avg_stats[task]["total_correct"] = task_correct
        task_avg_stats[task]["total_count"] = task_total
        task_avg_stats[task]["average_accuracy"] = task_correct / task_total if task_total > 0 else 0

    overall_accuracy = total_correct / total_count if total_count > 0 else 0
    return level2_stats, level3_stats, task_avg_stats, overall_accuracy, total_count

def main():
    parser = argparse.ArgumentParser(description="Evaluate accuracy with hierarchical categories.")
    parser.add_argument('jsonl_path', type=str, help="Path to input JSONL file.")
    args = parser.parse_args()

    data = load_jsonl_data(args.jsonl_path)
    level2_stats, level3_stats, task_avg_stats, overall_accuracy, total_count = calculate_accuracy(data)

    print("\n=== Level 2 (category + sub-category) Accuracy ===")
    for task, subcats in level2_stats.items():
        for subcat, stats in subcats.items():
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"[{task} / {subcat}] Accuracy: {acc:.4f} ({stats['correct']}/{stats['total']})")

    print("\n=== Level 3 (category + sub-category + sub-sub-category) Accuracy ===")
    for task, subcats in level3_stats.items():
        for subcat, subsubcats in subcats.items():
            for subsubcat, stats in subsubcats.items():
                acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                print(f"[{task} / {subcat} / {subsubcat}] Accuracy: {acc:.4f} ({stats['correct']}/{stats['total']})")

    print("\n=== Average Accuracy per Category ===")
    for task, stats in task_avg_stats.items():
        acc = stats["average_accuracy"]
        print(f"[{task}] Average Accuracy: {acc:.4f} ({stats['total_correct']}/{stats['total_count']})")

    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    print(f"Total count: {total_count}")

if __name__ == "__main__":
    # usage: python evaluate.py input.jsonl
    main()
