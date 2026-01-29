import csv
import matplotlib.pyplot as plt
from matplotlib import rcParams
import platform
import os
import numpy as np


def evaluate_and_plot_plate_recognition(csv_path):
    # 1. Initial setup and font configuration
    if not os.path.exists(csv_path):
        print(f"Error: File not found '{csv_path}'")
        return

    try:
        if platform.system() == 'Windows':
            rcParams['font.sans-serif'] = ['SimHei']
        elif platform.system() == 'Darwin':
            rcParams['font.sans-serif'] = ['Arial Unicode MS']
        else:
            rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
        rcParams['axes.unicode_minus'] = False
    except:
        print("Warning: Font configuration failed")

    # 2. Initialize data structures for per-plate metrics
    plate_metrics = []
    invalid_rows = 0

    # 3. Process CSV file
    with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header

        current_plate = None
        plate_data = None

        for row_idx, row in enumerate(reader, start=2):
            if len(row) < 18:
                print(f"[Row {row_idx}] ⚠️ Insufficient columns ({len(row)}), skipping")
                invalid_rows += 1
                continue

            plate_id = row[0]  # Assuming first column is plate ID
            if plate_id != current_plate:
                if current_plate is not None:
                    # Calculate metrics for previous plate
                    calculate_plate_metrics(plate_data)
                    plate_metrics.append(plate_data)

                # Start new plate
                current_plate = plate_id
                plate_data = {
                    'plate_id': plate_id,
                    'TP': 0, 'FP': 0, 'FN': 0, 'total_chars': 0, 'correct_chars': 0,
                    'TP_zh': 0, 'FP_zh': 0, 'FN_zh': 0, 'total_chars_zh': 0, 'correct_chars_zh': 0,
                    'TP_st': 0, 'FP_st': 0, 'FN_st': 0, 'total_chars_st': 0, 'correct_chars_st': 0
                }

            # Process Chinese character
            a_zh = row[2]
            p_zh = row[10]

            if a_zh and p_zh and a_zh == p_zh:
                plate_data['TP_zh'] += 1
                plate_data['correct_chars_zh'] += 1
            elif p_zh and not a_zh:
                plate_data['FP_zh'] += 1
            elif len(p_zh) > len(a_zh):
                plate_data['FP_zh'] += 1
            elif a_zh and not p_zh:
                plate_data['FN_zh'] += 1
            elif a_zh and p_zh and a_zh != p_zh:
                plate_data['FP_zh'] += 1
                plate_data['FN_zh'] += 1

            plate_data['total_chars_zh'] += 1

            # Process string characters
            actual = row[3:9]
            predicted = row[11:17]

            actual_8th = row[9].strip() if len(row) > 9 else ''
            predicted_8th = row[17].strip() if len(row) > 17 else ''
            if actual_8th or predicted_8th:
                actual.append(actual_8th)
                predicted.append(predicted_8th)

            length = max(len(actual), len(predicted))

            for i in range(length):
                a_st = str(actual[i]).strip() if i < len(actual) else ''
                p_st = str(predicted[i]).strip() if i < len(predicted) else ''

                if i == 7 and a_st == '' and p_st == '':
                    continue

                if a_st and p_st and a_st == p_st:
                    plate_data['TP_st'] += 1
                    plate_data['correct_chars_st'] += 1
                elif p_st and not a_st:
                    plate_data['FP_st'] += 1
                elif len(p_st) > len(a_st):
                    plate_data['FP_st'] += 1
                elif a_st and not p_st:
                    plate_data['FN_st'] += 1
                elif a_st and p_st and a_st != p_st:
                    plate_data['FP_st'] += 1
                    plate_data['FN_st'] += 1

                plate_data['total_chars_st'] += 1

        # Add the last plate
        if current_plate is not None:
            calculate_plate_metrics(plate_data)
            plate_metrics.append(plate_data)

    # 4. Calculate overall metrics
    overall_metrics = calculate_overall_metrics(plate_metrics)

    # 5. Print results
    print_results(overall_metrics, invalid_rows)

    # 6. Create visualizations
    plot_plate_metrics(plate_metrics)
    plot_overall_metrics(overall_metrics)


def calculate_plate_metrics(plate_data):
    """Calculate metrics for a single plate"""
    # Calculate overall metrics for the plate
    plate_data['TP'] = plate_data['TP_zh'] + plate_data['TP_st']
    plate_data['FP'] = plate_data['FP_zh'] + plate_data['FP_st']
    plate_data['FN'] = plate_data['FN_zh'] + plate_data['FN_st']
    plate_data['total_chars'] = plate_data['total_chars_zh'] + plate_data['total_chars_st']
    plate_data['correct_chars'] = plate_data['correct_chars_zh'] + plate_data['correct_chars_st']

    # Calculate all metrics
    plate_data['accuracy'] = plate_data['correct_chars'] / plate_data['total_chars'] if plate_data['total_chars'] else 0
    plate_data['precision'] = plate_data['TP'] / (plate_data['TP'] + plate_data['FP']) if (
                plate_data['TP'] + plate_data['FP']) else 0
    plate_data['recall'] = plate_data['TP'] / (plate_data['TP'] + plate_data['FN']) if (
                plate_data['TP'] + plate_data['FN']) else 0
    plate_data['f1'] = 2 * plate_data['precision'] * plate_data['recall'] / (
                plate_data['precision'] + plate_data['recall']) if (
                plate_data['precision'] + plate_data['recall']) else 0

    plate_data['accuracy_zh'] = plate_data['correct_chars_zh'] / plate_data['total_chars_zh'] if plate_data[
        'total_chars_zh'] else 0
    plate_data['precision_zh'] = plate_data['TP_zh'] / (plate_data['TP_zh'] + plate_data['FP_zh']) if (
                plate_data['TP_zh'] + plate_data['FP_zh']) else 0
    plate_data['recall_zh'] = plate_data['TP_zh'] / (plate_data['TP_zh'] + plate_data['FN_zh']) if (
                plate_data['TP_zh'] + plate_data['FN_zh']) else 0
    plate_data['f1_zh'] = 2 * plate_data['precision_zh'] * plate_data['recall_zh'] / (
                plate_data['precision_zh'] + plate_data['recall_zh']) if (
                plate_data['precision_zh'] + plate_data['recall_zh']) else 0

    plate_data['accuracy_st'] = plate_data['correct_chars_st'] / plate_data['total_chars_st'] if plate_data[
        'total_chars_st'] else 0
    plate_data['precision_st'] = plate_data['TP_st'] / (plate_data['TP_st'] + plate_data['FP_st']) if (
                plate_data['TP_st'] + plate_data['FP_st']) else 0
    plate_data['recall_st'] = plate_data['TP_st'] / (plate_data['TP_st'] + plate_data['FN_st']) if (
                plate_data['TP_st'] + plate_data['FN_st']) else 0
    plate_data['f1_st'] = 2 * plate_data['precision_st'] * plate_data['recall_st'] / (
                plate_data['precision_st'] + plate_data['recall_st']) if (
                plate_data['precision_st'] + plate_data['recall_st']) else 0


def calculate_overall_metrics(plate_metrics):
    """Calculate overall metrics across all plates"""
    overall = {
        'TP': sum(p['TP'] for p in plate_metrics),
        'FP': sum(p['FP'] for p in plate_metrics),
        'FN': sum(p['FN'] for p in plate_metrics),
        'total_chars': sum(p['total_chars'] for p in plate_metrics),
        'correct_chars': sum(p['correct_chars'] for p in plate_metrics),
        'TP_zh': sum(p['TP_zh'] for p in plate_metrics),
        'FP_zh': sum(p['FP_zh'] for p in plate_metrics),
        'FN_zh': sum(p['FN_zh'] for p in plate_metrics),
        'total_chars_zh': sum(p['total_chars_zh'] for p in plate_metrics),
        'correct_chars_zh': sum(p['correct_chars_zh'] for p in plate_metrics),
        'TP_st': sum(p['TP_st'] for p in plate_metrics),
        'FP_st': sum(p['FP_st'] for p in plate_metrics),
        'FN_st': sum(p['FN_st'] for p in plate_metrics),
        'total_chars_st': sum(p['total_chars_st'] for p in plate_metrics),
        'correct_chars_st': sum(p['correct_chars_st'] for p in plate_metrics)
    }

    # Calculate overall metrics
    overall['accuracy'] = overall['correct_chars'] / overall['total_chars'] if overall['total_chars'] else 0
    overall['precision'] = overall['TP'] / (overall['TP'] + overall['FP']) if (overall['TP'] + overall['FP']) else 0
    overall['recall'] = overall['TP'] / (overall['TP'] + overall['FN']) if (overall['TP'] + overall['FN']) else 0
    overall['f1'] = 2 * overall['precision'] * overall['recall'] / (overall['precision'] + overall['recall']) if (
                overall['precision'] + overall['recall']) else 0

    overall['accuracy_zh'] = overall['correct_chars_zh'] / overall['total_chars_zh'] if overall['total_chars_zh'] else 0
    overall['precision_zh'] = overall['TP_zh'] / (overall['TP_zh'] + overall['FP_zh']) if (
                overall['TP_zh'] + overall['FP_zh']) else 0
    overall['recall_zh'] = overall['TP_zh'] / (overall['TP_zh'] + overall['FN_zh']) if (
                overall['TP_zh'] + overall['FN_zh']) else 0
    overall['f1_zh'] = 2 * overall['precision_zh'] * overall['recall_zh'] / (
                overall['precision_zh'] + overall['recall_zh']) if (
                overall['precision_zh'] + overall['recall_zh']) else 0

    overall['accuracy_st'] = overall['correct_chars_st'] / overall['total_chars_st'] if overall['total_chars_st'] else 0
    overall['precision_st'] = overall['TP_st'] / (overall['TP_st'] + overall['FP_st']) if (
                overall['TP_st'] + overall['FP_st']) else 0
    overall['recall_st'] = overall['TP_st'] / (overall['TP_st'] + overall['FN_st']) if (
                overall['TP_st'] + overall['FN_st']) else 0
    overall['f1_st'] = 2 * overall['precision_st'] * overall['recall_st'] / (
                overall['precision_st'] + overall['recall_st']) if (
                overall['precision_st'] + overall['recall_st']) else 0

    return overall


def print_results(overall_metrics, invalid_rows):
    """Print the evaluation results"""
    print("\n===== 车牌字符识别性能指标 =====")
    print(f"Invalid data rows: {invalid_rows}")
    print(f"Total number of characters: {overall_metrics['total_chars']}")
    print(f"Correct number of characters: {overall_metrics['correct_chars']}")
    print(f"TP of total characters: {overall_metrics['TP']}")
    print(f"FP of total characters: {overall_metrics['FP']}")
    print(f"FN of total characters: {overall_metrics['FN']}")
    print("-------------------------------")
    print(f"Accuracy of total characters: {overall_metrics['accuracy']:.4f}")
    print(f"Precision of total characters: {overall_metrics['precision']:.4f}")
    print(f"Recall of total characters: {overall_metrics['recall']:.4f}")
    print(f"F1 Score of total characters: {overall_metrics['f1']:.4f}")
    print("-------------------------------")
    print("-------------------------------")
    print(f"Total number of Chinese characters: {overall_metrics['total_chars_zh']}")
    print(f"The correct number of Chinese characters: {overall_metrics['correct_chars_zh']}")
    print(f"TP of Chinese characters: {overall_metrics['TP_zh']}")
    print(f"FP of Chinese characters: {overall_metrics['FP_zh']}")
    print(f"FN of Chinese characters: {overall_metrics['FN_zh']}")
    print("-------------------------------")
    print(f"Accuracy of Chinese characters: {overall_metrics['accuracy_zh']:.4f}")
    print(f"Precision of Chinese characters: {overall_metrics['precision_zh']:.4f}")
    print(f"Recall of Chinese characters: {overall_metrics['recall_zh']:.4f}")
    print(f"F1 Score of Chinese characters: {overall_metrics['f1_zh']:.4f}")
    print("-------------------------------")
    print("-------------------------------")
    print(f"The total number of letters and numbers: {overall_metrics['total_chars_st']}")
    print(f"The correct number of letters and numbers: {overall_metrics['correct_chars_st']}")
    print(f"TP of letters and numbers: {overall_metrics['TP_st']}")
    print(f"FP of letters and numbers: {overall_metrics['FP_st']}")
    print(f"FN of letters and numbers: {overall_metrics['FN_st']}")
    print("-------------------------------")
    print(f"Accuracy of letters and numbers: {overall_metrics['accuracy_st']:.4f}")
    print(f"Precision of letters and numbers: {overall_metrics['precision_st']:.4f}")
    print(f"Recall of letters and numbers: {overall_metrics['recall_st']:.4f}")
    print(f"F1 Score of letters and numbers: {overall_metrics['f1_st']:.4f}")


def plot_plate_metrics(plate_metrics):
    """Plot per-plate metrics over all plates"""
    plt.figure(figsize=(15, 12))

    # Prepare data for plotting
    plate_ids = [f"{i + 1}" for i in range(len(plate_metrics))]
    accuracies = [p['accuracy'] for p in plate_metrics]
    precisions = [p['precision'] for p in plate_metrics]
    recalls = [p['recall'] for p in plate_metrics]
    f1_scores = [p['f1'] for p in plate_metrics]

    accuracies_zh = [p['accuracy_zh'] for p in plate_metrics]
    precisions_zh = [p['precision_zh'] for p in plate_metrics]
    recalls_zh = [p['recall_zh'] for p in plate_metrics]
    f1_scores_zh = [p['f1_zh'] for p in plate_metrics]

    accuracies_st = [p['accuracy_st'] for p in plate_metrics]
    precisions_st = [p['precision_st'] for p in plate_metrics]
    recalls_st = [p['recall_st'] for p in plate_metrics]
    f1_scores_st = [p['f1_st'] for p in plate_metrics]

    # Plot 1: Overall metrics trend
    plt.subplot(3, 1, 1)
    plt.plot(plate_ids, accuracies, label='Accuracy', marker='o')
    plt.plot(plate_ids, precisions, label='Precision', marker='s')
    plt.plot(plate_ids, recalls, label='Recall', marker='^')
    plt.plot(plate_ids, f1_scores, label='F1 Score', marker='d')
    plt.ylim(0, 1.1)
    plt.title('Overall Metrics Trend by Plate')
    plt.ylabel('Score')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Chinese character metrics trend
    plt.subplot(3, 1, 2)
    plt.plot(plate_ids, accuracies_zh, label='Accuracy (Chinese)', marker='o')
    plt.plot(plate_ids, precisions_zh, label='Precision (Chinese)', marker='s')
    plt.plot(plate_ids, recalls_zh, label='Recall (Chinese)', marker='^')
    plt.plot(plate_ids, f1_scores_zh, label='F1 Score (Chinese)', marker='d')
    plt.ylim(0, 1.1)
    plt.title('Chinese Character Metrics Trend by Plate')
    plt.ylabel('Score')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot 3: String metrics trend
    plt.subplot(3, 1, 3)
    plt.plot(plate_ids, accuracies_st, label='Accuracy (String)', marker='o')
    plt.plot(plate_ids, precisions_st, label='Precision (String)', marker='s')
    plt.plot(plate_ids, recalls_st, label='Recall (String)', marker='^')
    plt.plot(plate_ids, f1_scores_st, label='F1 Score (String)', marker='d')
    plt.ylim(0, 1.1)
    plt.title('String Metrics Trend by Plate')
    plt.ylabel('Score')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_overall_metrics(overall_metrics):
    """Plot the overall metrics dashboard"""
    plt.figure(figsize=(18, 15))

    # Plot 1: Main metrics
    plt.subplot(4, 3, 1)
    main_metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    values = [overall_metrics['accuracy'], overall_metrics['precision'],
              overall_metrics['recall'], overall_metrics['f1']]
    bars = plt.bar(main_metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylim(0, 1.1)
    plt.title('Main Evaluation Metrics')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}', ha='center', va='bottom')

    # Plot 2-5: Metric breakdowns
    metric_plots = [
        ('Accuracy', ['Overall', 'Chinese', 'String'],
         [overall_metrics['accuracy'], overall_metrics['accuracy_zh'], overall_metrics['accuracy_st']]),
        ('Precision', ['Overall', 'Chinese', 'String'],
         [overall_metrics['precision'], overall_metrics['precision_zh'], overall_metrics['precision_st']]),
        ('Recall', ['Overall', 'Chinese', 'String'],
         [overall_metrics['recall'], overall_metrics['recall_zh'], overall_metrics['recall_st']]),
        ('F1 Score', ['Overall', 'Chinese', 'String'],
         [overall_metrics['f1'], overall_metrics['f1_zh'], overall_metrics['f1_st']])
    ]

    for i, (title, labels, values) in enumerate(metric_plots, 2):
        plt.subplot(4, 3, i)
        bars = plt.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.ylim(0, 1.1)
        plt.title(f'{title} Comparison')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}', ha='center', va='bottom')

    # Plot 6-11: Count breakdowns
    count_plots = [
        ('Error Types', ['TP', 'FP', 'FN'], [overall_metrics['TP'], overall_metrics['FP'], overall_metrics['FN']],
         '#1f77b4'),
        ('True Positives', ['Overall', 'Chinese', 'String'],
         [overall_metrics['TP'], overall_metrics['TP_zh'], overall_metrics['TP_st']],
         '#2ca02c'),
        ('False Positives', ['Overall', 'Chinese', 'String'],
         [overall_metrics['FP'], overall_metrics['FP_zh'], overall_metrics['FP_st']],
         '#d62728'),
        ('False Negatives', ['Overall', 'Chinese', 'String'],
         [overall_metrics['FN'], overall_metrics['FN_zh'], overall_metrics['FN_st']],
         '#ff7f0e'),
        ('Character Counts', ['Total', 'Chinese', 'String'],
         [overall_metrics['total_chars'], overall_metrics['total_chars_zh'], overall_metrics['total_chars_st']],
         '#9467bd'),
        ('Correct Characters', ['Overall', 'Chinese', 'String'],
         [overall_metrics['correct_chars'], overall_metrics['correct_chars_zh'], overall_metrics['correct_chars_st']],
         '#8c564b')
    ]

    for i, (title, labels, values, color) in enumerate(count_plots, 6):
        plt.subplot(4, 3, i)
        max_val = max(values) if values else 1
        bars = plt.bar(labels, values, color=color)
        plt.ylim(0, max_val * 1.2)
        plt.title(title)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# Run the evaluation
evaluate_and_plot_plate_recognition('label.csv')