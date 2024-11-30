import argparse


parser = argparse.ArgumentParser(description='GPT Language Model')
parser.add_argument('--path', type=str, help='Path to the input file')
parser.add_argument('--pitch-path', type=str, help='Path to the pitch file')
parser.add_argument('--timing-path', type=str, help='Path to the timing file')
parser.add_argument('--interleaved-path', type=str, help='Path to the interleaved file')
parser.add_argument('--unaltered-path', type=str, help='Path to the unaltered file')
args = parser.parse_args()

with open(args.path, 'r') as file:
    input_text = file.read()
    
pitch_data = ''
timing_data = ''
interleaved_data = ''
count = 1
last_char = None
for char in input_text:
    if last_char:
        if char == last_char:
            pitch_data += ''
            count += 1
        else:
            pitch_data += char
            timing_data += str(count)
            interleaved_data += 'T' + str(count) + 'C' + char
            count = 1
    last_char = char


with open(args.pitch_path, 'w') as pitch_file:
    pitch_file.write(pitch_data)
    
    
with open(args.timing_path, 'w') as timing_file:
    timing_file.write(timing_data)
    
with open(args.unaltered_path, 'w') as unaltered_file:
    unaltered_file.write(input_text)
    
with open(args.interleaved_path, 'w') as interleaved_file:
    interleaved_file.write(interleaved_data)
