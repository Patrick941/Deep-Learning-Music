import argparse
import csv


parser = argparse.ArgumentParser(description='Compare output files and generate a CSV report.')
parser.add_argument('--truth', required=True, help='Path to the truth file')
parser.add_argument('--output-1', required=True, help='Path to the first output file')
parser.add_argument('--output-2', required=True, help='Path to the second output file')
parser.add_argument('--output-3', required=True, help='Path to the third output file')
parser.add_argument('--output-csv', required=True, help='Path to the output CSV file')
args = parser.parse_args()

csv_headers = ['File', 'Character Distribution Distance', 'Pattern Distribution Distance']
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
    i = 0
    while i < len(data) - 2:
        pattern = data[i:i+3]
        if pattern[0] == pattern[1] == pattern[2]:
            if pattern in patterns:
                patterns[pattern] += 1
            else:
                patterns[pattern] = 1
            i += 3
        else:
            i += 1
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
    return lengths
        

truth_data = ''
with open(args.truth, 'r') as file:
    truth_data = file.read()
    truth_char_distribution = get_char_distribution(truth_data)
    truth_patterns = get_patterns(truth_data)
    truth_sequence_lengths = get_sequence_lengths(truth_data)

output_files = [args.output_1, args.output_2, args.output_3]
for output_file in output_files:
    with open(output_file, 'r') as file:
        model_data = file.read()
        model_char_distribution = get_char_distribution(model_data)
        char_distance = 0
        for char in truth_char_distribution:
            truth_freq = truth_char_distribution.get(char, 0)
            model_freq = model_char_distribution.get(char, 0)
            char_distance += abs(truth_freq - model_freq)
            
        pattern_distance = 0
        model_patterns = get_patterns(model_data)
        for pattern in truth_patterns:
            truth_freq = truth_patterns.get(pattern, 0)
            model_freq = model_patterns.get(pattern, 0)
            pattern_distance += abs(truth_freq - model_freq)
            
        lengths_distance = 0
        model_sequence_lengths = get_sequence_lengths(model_data)
        for length in truth_sequence_lengths:
            truth_freq = truth_sequence_lengths.get(length, 0)
            model_freq = model_sequence_lengths.get(length, 0)
            lengths_distance += abs(truth_freq - model_freq)

    with open(args.output_csv, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([output_file, char_distance, pattern_distance])
        
    
    