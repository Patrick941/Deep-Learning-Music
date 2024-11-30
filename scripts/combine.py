import argparse

parser = argparse.ArgumentParser(description='Combine pitch and timing files into an output file.')
parser.add_argument('--pitch', required=True, help='Path to the pitch file')
parser.add_argument('--timing', required=True, help='Path to the timing file')
parser.add_argument('--output', required=True, help='Path to the output file')

args = parser.parse_args()

with open(args.pitch, 'r') as pitch_file:
    pitch_data = pitch_file.read()
    
with open(args.timing, 'r') as timing_file:
    timing_data = timing_file.read()
    
output_data = ''
    
while True:
    output_data += (pitch_data[0] * int(timing_data[0]))
    pitch_data = pitch_data[1:]
    timing_data = timing_data[1:]
    if not pitch_data or not timing_data:
        break

with open(args.output, 'w') as output_file:
    output_file.write(output_data)