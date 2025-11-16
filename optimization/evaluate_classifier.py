"""Evaluation script for Query Type Classifier."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import time

from query_type_classifier import QueryTypeClassifier, TASK_TYPES


def evaluate_classifier(
    eval_data_path: str,
    model_name: str = 'Qwen/Qwen3-4B-Instruct-2507',
    output_path: str = None,
    verbose: bool = True
) -> Dict:
    """
    Evaluate the QueryTypeClassifier on the evaluation dataset.
    
    Args:
        eval_data_path: Path to the JSON file containing evaluation data
        model_name: Model name for classification
        output_path: Optional path to save detailed results JSON
        verbose: Whether to print progress and results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load evaluation data
    if verbose:
        print(f"Loading evaluation data from {eval_data_path}...")
    
    with open(eval_data_path, 'r') as f:
        eval_data = json.load(f)
    
    if verbose:
        print(f"Loaded {len(eval_data)} questions")
    
    # Initialize classifier
    if verbose:
        print(f"Initializing classifier with model: {model_name}...")
    classifier = QueryTypeClassifier(model_name=model_name)
    
    # Evaluate each question
    results = []
    correct = 0
    total = len(eval_data)
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    start_time = time.time()
    
    # or_questions = [q for q in eval_data if q['task_type'] == 'Object Reasoning']
    # ar_questions = [q for q in eval_data if q['task_type'] == 'Action Reasoning']

    # # Print them side-by-side for manual inspection
    # for q in or_questions[:15]:
    #     print(f"OR: {q['question']}")
    # for q in ar_questions[:15]:
    #     print(f"AR: {q['question']}")

    for i, item in enumerate(eval_data):
        question = item['question']
        ground_truth = item['task_type']
        question_id = item.get('question_id', f"item_{i}")
        
        # Classify
        try:
            predicted = classifier.classify(question)
            is_correct = (predicted == ground_truth)
            
            if is_correct:
                correct += 1
                per_class_correct[ground_truth] += 1
            
            per_class_total[ground_truth] += 1
            confusion_matrix[ground_truth][predicted] += 1
            
            results.append({
                'question_id': question_id,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'correct': is_correct
            })
            
            if verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = (total - i - 1) * avg_time
                print(f"Processed {i + 1}/{total} questions "
                      f"(Accuracy: {correct/(i+1):.2%}, "
                      f"ETA: {remaining/60:.1f} min)")
        
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            results.append({
                'question_id': question_id,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': 'ERROR',
                'correct': False,
                'error': str(e)
            })
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    
    per_class_accuracy = {}
    for task_type in TASK_TYPES:
        if per_class_total[task_type] > 0:
            per_class_accuracy[task_type] = per_class_correct[task_type] / per_class_total[task_type]
        else:
            per_class_accuracy[task_type] = 0.0
    
    # Build summary
    summary = {
        'total_questions': total,
        'correct': correct,
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
        'per_class_counts': dict(per_class_total),
        'confusion_matrix': {k: dict(v) for k, v in confusion_matrix.items()},
        'total_time_seconds': total_time,
        'avg_time_per_question': total_time / total if total > 0 else 0.0
    }
    
    # Print results
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total Questions: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"\nTotal Time: {total_time/60:.2f} minutes")
        print(f"Average Time per Question: {total_time/total:.2f} seconds")
        
        print("\n" + "-"*60)
        print("Per-Class Accuracy:")
        print("-"*60)
        for task_type in sorted(per_class_accuracy.keys()):
            count = per_class_total[task_type]
            acc = per_class_accuracy[task_type]
            print(f"  {task_type:25s} {acc:6.2%} ({per_class_correct[task_type]}/{count})")
        
        print("\n" + "-"*60)
        print("Confusion Matrix (Top 10 most common errors):")
        print("-"*60)
        errors = []
        for true_type, pred_dict in confusion_matrix.items():
            for pred_type, count in pred_dict.items():
                if true_type != pred_type and count > 0:
                    errors.append((true_type, pred_type, count))
        
        errors.sort(key=lambda x: x[2], reverse=True)
        for true_type, pred_type, count in errors[:10]:
            print(f"  {true_type:25s} -> {pred_type:25s} ({count} times)")
    
    # Save detailed results if requested
    if output_path:
        output_data = {
            'summary': summary,
            'detailed_results': results
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        if verbose:
            print(f"\nDetailed results saved to {output_path}")
    
    return {
        'summary': summary,
        'detailed_results': results
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Query Type Classifier')
    parser.add_argument(
        '--eval_data',
        type=str,
        default='data/eval_subset.json',
        help='Path to evaluation data JSON file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-4B-Instruct-2507',
        help='Model name for classification'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional path to save detailed results JSON'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    evaluate_classifier(
        eval_data_path=args.eval_data,
        model_name=args.model,
        output_path=args.output,
        verbose=not args.quiet
    )

