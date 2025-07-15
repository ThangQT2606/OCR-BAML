import os
import cv2
import json
import time
from engine_capcha import init_cr, llm_predict
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import uuid


def get_ground_truth_from_filename(filename):
    """Extract ground truth label from filename (without extension)"""
    return os.path.splitext(filename)[0]


def run_evaluation(test_folder="test", llm_model="Gemini_2_0_pro"):
    """Run captcha recognition evaluation on all images in test folder"""
    
    # Initialize client registry
    cr = init_cr(LLM_CR=llm_model)
    
    # Get all image files in test folder
    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images in {test_folder} folder")
    
    results = []
    predictions = []
    ground_truths = []
    
    for i, filename in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {filename}")
        
        # Get ground truth from filename
        ground_truth = get_ground_truth_from_filename(filename)
        ground_truths.append(ground_truth)
        
        # Load image
        image_path = os.path.join(test_folder, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image {image_path}")
            predictions.append("")
            continue
        
        # Predict using LLM
        try:
            uid = str(uuid.uuid1())
            start_time = time.time()
            result = llm_predict(uuid=uid, files_name=filename, images=[image], cr=cr)
            end_time = time.time()
            
            # Extract predicted content
            if result and len(result) > 0 and result[0]["extract_data"]:
                predicted_content = result[0]["extract_data"].get("content", "")
            else:
                predicted_content = ""
            
            predictions.append(predicted_content)
            
            # Store detailed result
            results.append({
                "filename": filename,
                "ground_truth": ground_truth,
                "predicted": predicted_content,
                "correct": ground_truth.lower() == predicted_content.lower(),
                "processing_time": end_time - start_time,
                "tokens": result[0]["tokens"] if result and len(result) > 0 else [0, 0]
            })
            
            print(f"  Ground Truth: {ground_truth}")
            print(f"  Predicted: {predicted_content}")
            print(f"  Correct: {ground_truth.lower() == predicted_content.lower()}")
            print(f"  Time: {end_time - start_time:.2f}s")
            print()
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            predictions.append("")
            results.append({
                "filename": filename,
                "ground_truth": ground_truth,
                "predicted": "",
                "correct": False,
                "processing_time": 0,
                "tokens": [0, 0],
                "error": str(e)
            })
    
    return results, predictions, ground_truths


def calculate_metrics(results, predictions, ground_truths):
    """Calculate accuracy and other metrics"""
    
    # Case-insensitive comparison
    correct_predictions = [pred.lower() for pred in predictions]
    correct_ground_truths = [gt.lower() for gt in ground_truths]
    
    # Calculate accuracy
    accuracy = accuracy_score(correct_ground_truths, correct_predictions)
    
    # Calculate character-level accuracy for partially correct predictions
    char_accuracies = []
    for gt, pred in zip(ground_truths, predictions):
        if len(gt) > 0:
            char_acc = sum(1 for a, b in zip(gt.lower(), pred.lower()) if a == b) / len(gt)
            char_accuracies.append(char_acc)
    
    avg_char_accuracy = np.mean(char_accuracies) if char_accuracies else 0
    
    return {
        "accuracy": accuracy,
        "character_accuracy": avg_char_accuracy,
        "total_samples": len(ground_truths),
        "correct_samples": sum(1 for r in results if r["correct"])
    }


def plot_confusion_matrix(ground_truths, predictions, save_path="confusion_matrix.png"):
    """Plot confusion matrix for unique character combinations"""
    
    # Case-insensitive comparison
    y_true = [gt.lower() for gt in ground_truths]
    y_pred = [pred.lower() for pred in predictions]
    
    # Get unique labels
    unique_labels = sorted(list(set(y_true + y_pred)))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Plot with smaller, more reasonable size
    fig_width = min(max(8, len(unique_labels) * 0.5), 15)
    fig_height = min(max(6, len(unique_labels) * 0.4), 12)
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix - Captcha Recognition')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Confusion matrix saved to {save_path}")


def save_detailed_results(llm_model, results, metrics, output_file="captcha_evaluation_results.json"):
    """Save detailed results to JSON file"""
    
    output_data = {
        "llm_model": llm_model,
        "metrics": metrics,
        "detailed_results": results,
        "summary": {
            "total_images": len(results),
            "correct_predictions": sum(1 for r in results if r["correct"]),
            "total_tokens": sum(r["tokens"][0] + r["tokens"][1] for r in results if "tokens" in r),
            "total_processing_time": sum(r["processing_time"] for r in results),
            "avg_processing_time": np.mean([r["processing_time"] for r in results])
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed results saved to {output_file}")


def print_summary(metrics, results):
    """Print evaluation summary"""
    
    print("=" * 60)
    print("CAPTCHA RECOGNITION EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Correct Predictions: {metrics['correct_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Character-level Accuracy: {metrics['character_accuracy']:.4f} ({metrics['character_accuracy']*100:.2f}%)")
    
    if results:
        total_time = sum(r["processing_time"] for r in results)
        avg_time = np.mean([r["processing_time"] for r in results])
        total_tokens = sum(r["tokens"][0] + r["tokens"][1] for r in results if "tokens" in r)
        
        print(f"Total Processing Time: {total_time:.2f} seconds")
        print(f"Average Processing Time: {avg_time:.2f} seconds per image")
        print(f"Total Tokens Used: {total_tokens}")
    
    print("=" * 60)


def main(llm_model: str, out: str):
    """Main evaluation function"""
    
    print("Starting Captcha Recognition Evaluation...")
    print()
    
    # Run evaluation
    results, predictions, ground_truths = run_evaluation(llm_model=llm_model)
    
    # Calculate metrics
    metrics = calculate_metrics(results, predictions, ground_truths)
    
    # Print summary
    print_summary(metrics, results)
    
    # Save detailed results
    save_detailed_results(llm_model, results, metrics, out)
    
    # Plot confusion matrix
    plot_confusion_matrix(ground_truths, predictions)
    
    # Show some examples of incorrect predictions
    print("\nIncorrect Predictions (first 10):")
    print("-" * 40)
    incorrect_count = 0
    for result in results:
        if not result["correct"] and incorrect_count < 10:
            print(f"File: {result['filename']}")
            print(f"  Expected: {result['ground_truth']}")
            print(f"  Got: {result['predicted']}")
            print()
            incorrect_count += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate captcha recognition")
    parser.add_argument("--llm_model", type=str, default="Gemini_2_0_flash", help="LLM model to use")
    parser.add_argument("--out", type=str, default="captcha_evaluation_results.json", help="Path save results")
    args = parser.parse_args()
    main(args.llm_model, args.out)