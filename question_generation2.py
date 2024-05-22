from transformers import pipeline, set_seed
import torch

def generate_questions(text, model="valhalla/t5-small-qg-prepend", max_length=64, num_return_sequences=1):
    question_generator = pipeline('text2text-generation', model=model)
    set_seed(42)
    questions = question_generator(text, max_length=max_length, num_return_sequences=num_return_sequences)
    return [q['generated_text'] for q in questions]

def main():
    file_path = 'path_to_your_file.txt'

    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split the text into smaller passages if needed
    passages = text.split('\n\n')  # Example split by empty lines

    for passage in passages:
        print(f"Passage: {passage[:50]}...")  # Print a part of the passage for reference
        questions = generate_questions(passage)
        for question in questions:
            print(f"Generated Question: {question}")

if __name__ == "__main__":
    main()
