import argparse

parser = argparse.ArgumentParser(description='GPT Language Model')
parser.add_argument('--path', type=str, help='Path to the input file')
args = parser.parse_args()

data = ''

with open(args.path, 'r') as file:
    input_text = file.read()
    
separated_text = input_text.split('T')
for text in separated_text:
    if text:
        number = ''.join(filter(str.isdigit, text))
        final_character = text[-1]
        data += final_character * int(number)
        
with open(args.path, 'w') as file:
    file.write(data)