import argparse
import csv
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Compare output files and generate a CSV report.')
parser.add_argument('--truth', required=True, help='Path to the truth file')
parser.add_argument('--outputs', required=True, nargs='+', help='Paths to the output files')
parser.add_argument('--output-csv', required=True, help='Path to the output CSV file')
args = parser.parse_args()

csv_headers = ['File', 'Character Distribution Distance', 'Pattern Distribution Distance', 'Pattern Length Distribution Distance', 'Pattern Recall', 'Pattern Precision', 'Pattern F1 Score']
with open(args.output_csv, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(csv_headers)

def get_char_distribution(data):
    char_distribution = {}
    for char in data:
        if char in char_distribution:
            char_distribution[char] += 1
        else:
            char_distribution[char] = 1
    total_chars = sum(char_distribution.values())
    for char in char_distribution:
        char_distribution[char] /= total_chars
    return char_distribution

def get_patterns(data):
    patterns = {}
    for i in range(len(data)):
        for j in range(2, 4):
            if i + j <= len(data):
                pattern = data[i:i + j]
                if pattern in patterns:
                    patterns[pattern] += 1
                else:
                    patterns[pattern] = 1
    total_patterns = sum(patterns.values())
    for pattern in patterns:
        patterns[pattern] /= total_patterns
    return patterns 

def get_sequence_lengths(data):
    lengths = {}
    i = 0
    length = 1
    last_char = None
    for char in data:
        if last_char:
            if char == last_char:
                length += 1
            else:
                if length in lengths:
                    lengths[length] += 1
                else:
                    lengths[length] = 1
                length = 1
        last_char = char
    if length in lengths:
        lengths[length] += 1
    else:
        lengths[length] = 1
    total_lengths = sum(lengths.values())
    for length in lengths:
        lengths[length] /= total_lengths
    return lengths

truth_data = ''
with open(args.truth, 'r') as file:
    truth_data = file.read()
    truth_char_distribution = get_char_distribution(truth_data)
    truth_patterns = get_patterns(truth_data)
    truth_sequence_lengths = get_sequence_lengths(truth_data)
    
    results = {}

    for output_file in args.outputs:
        with open(output_file, 'r') as file:
            model_data = file.read()
            model_char_distribution = get_char_distribution(model_data)
            char_distance = 0
            for char in truth_char_distribution:
                truth_freq = truth_char_distribution.get(char, 0)
                model_freq = model_char_distribution.get(char, 0)
                char_distance += abs(truth_freq - model_freq)
            char_distance /= len(truth_char_distribution)
                
            pattern_distance = 0
            model_patterns = get_patterns(model_data)
            for pattern in truth_patterns:
                truth_freq = truth_patterns.get(pattern, 0)
                model_freq = model_patterns.get(pattern, 0)
                pattern_distance += abs(truth_freq - model_freq)
            pattern_distance /= len(truth_patterns)
                
            lengths_distance = 0
            model_sequence_lengths = get_sequence_lengths(model_data)
            for length in truth_sequence_lengths:
                truth_freq = truth_sequence_lengths.get(length, 0)
                model_freq = model_sequence_lengths.get(length, 0)
                lengths_distance += abs(truth_freq - model_freq)
            lengths_distance /= len(truth_sequence_lengths)
                
            pattern_count = 0
            for pattern in truth_patterns:
                if pattern in model_patterns:
                    pattern_count += 1
            if len(model_patterns) == 0:
                pattern_recall = 0
                pattern_precision = 0
                pattern_f1_score = 0
            else:
                pattern_recall = pattern_count / len(truth_patterns)
                pattern_precision = pattern_count / len(model_patterns)
                pattern_f1_score = 2 * (pattern_precision * pattern_recall) / (pattern_precision + pattern_recall)
                    
            output_file_name = output_file.split('/')[-1]
            results[output_file_name] = {
                'Character Distribution Distance': char_distance,
                'Pattern Distribution Distance': pattern_distance,
                'Pattern Length Distribution Distance': lengths_distance,
                'Pattern Recall': pattern_recall,
                'Pattern Precision': pattern_precision,
                'Pattern F1 Score': pattern_f1_score
            }

        with open(args.output_csv, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([output_file_name, f"{char_distance:.4f}", f"{pattern_distance:.4f}", f"{lengths_distance:.4f}", f"{pattern_recall:.4f}", f"{pattern_precision:.4f}", f"{pattern_f1_score:.4f}"])

for metric in csv_headers[1:]:
    plt.figure(figsize=(10, 5))
    values = []
    for key in results:
        values.append(results[key][metric])
    
    non_zero_keys = []
    non_zero_values = []
    for key, value in zip(results.keys(), values):
        if value != 0:
            non_zero_keys.append(key)
            non_zero_values.append(value)
    
    plt.bar(non_zero_keys, non_zero_values)
    plt.title(metric)
    plt.xlabel('Output Files')
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'images/{metric}.png')