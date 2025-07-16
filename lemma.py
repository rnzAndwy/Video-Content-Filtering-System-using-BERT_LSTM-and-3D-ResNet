import csv
import json
import argparse

def read_csv_column(file_path, column_name):
    """
    Read a specific column from a CSV file and return its unique values.
    """
    unique_words = set()
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if column_name not in reader.fieldnames:
            raise ValueError(f"Column '{column_name}' not found in the CSV file.")
        for row in reader:
            word = row[column_name].lower().strip()
            if word:
                unique_words.add(word)
    return unique_words

def create_lemmatizer_dict(words):
    """
    Create a dictionary for lemmatization based on the input words.
    """
    lemma_dict = {}
    for word in words:
        # Simple lemmatization: use the word as its own lemma
        lemma_dict[word] = word
        
        # Add common variations
        if word.endswith('s'):
            lemma_dict[word[:-1]] = word  # singular form
        if word.endswith('es'):
            lemma_dict[word[:-2]] = word  # singular form
        if word.endswith('ies'):
            lemma_dict[word[:-3] + 'y'] = word  # singular form
        if word.endswith('ing'):
            lemma_dict[word[:-3]] = word  # base form
            lemma_dict[word[:-3] + 'e'] = word  # base form with 'e'
        if word.endswith('ed'):
            lemma_dict[word[:-2]] = word  # base form
            lemma_dict[word[:-1]] = word  # base form with 'e'
    
    return lemma_dict

def save_as_json(data, file_path):
    """
    Save the dictionary as a JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def save_as_python_dict(data, file_path):
    """
    Save the dictionary as a Python file containing a dictionary.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("lemmatizer_dict = {\n")
        for key, value in data.items():
            f.write(f"    '{key}': '{value}',\n")
        f.write("}\n")

def main():
    parser = argparse.ArgumentParser(description="Convert CSV column to lemmatizer dictionary")
    parser.add_argument("csv_file", help=r'C:\Users\Renz\Desktop\Profanity Detection\profane1.csv')
    parser.add_argument("column_name", help="text")
    parser.add_argument("output_prefix", help="profanity_lemmatizer")
    args = parser.parse_args()

    try:
        words = read_csv_column(args.csv_file, args.column_name)
        lemma_dict = create_lemmatizer_dict(words)
        
        json_output = f"{args.output_prefix}_lemmatizer_dict.json"
        save_as_json(lemma_dict, json_output)
        print(f"JSON dictionary saved to {json_output}")
        
        python_output = f"{args.output_prefix}_lemmatizer_dict.py"
        save_as_python_dict(lemma_dict, python_output)
        print(f"Python dictionary saved to {python_output}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()