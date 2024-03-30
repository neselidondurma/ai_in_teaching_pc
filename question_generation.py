import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import json

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

def extract_key_phrases(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    key_phrases = set()

    for sentence in sentences:
        # Tokenize each sentence into words and tag part of speech
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)

        # Extract all noun phrases as key phrases
        for word, tag in tagged_words:
            if tag in ['NN', 'NNS', 'NNP', 'NNPS']:  # Nouns and Proper Nouns
                key_phrases.add(word.lower())

    # Remove stopwords from key phrases
    stop_words = set(stopwords.words('german'))
    key_phrases = [phrase for phrase in key_phrases if phrase not in stop_words]

    return key_phrases

def formulate_questions(key_phrases):
    questions = []
    for phrase in key_phrases:
        # Example templates for questions
        questions.append(f"Was ist {phrase}?")
        questions.append(f"Welche Rolle spielt {phrase} in der Thermodynamik?")
    return questions

def extract_answers(text, key_phrases):
    sentences = sent_tokenize(text)
    answers = {}

    for phrase in key_phrases:
        for sentence in sentences:
            if phrase in sentence.lower():
                # Assign the sentence as an answer to the phrase
                answers[phrase] = sentence
                break

    return answers

def main():
    file_path = 'chapter_1.txt'

    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Extract key phrases from the text
    key_phrases = extract_key_phrases(text)

    # Formulate questions based on key phrases
    questions = formulate_questions(key_phrases)

    # Extract/summarize answers from the text
    answers = extract_answers(text, key_phrases)

    # Output question-answer pairs
    for question in questions:
        key_phrase = question.split()[2].rstrip('?')  # Simple way to get the key phrase back from the question
        answer = answers.get(key_phrase, "No answer found.")
        print(f"Q: {question}\nA: {answer}\n")
    # Create a list of dictionaries for each QA pair
    qa_pairs = [{"question": question, "answer": answers.get(question.split()[2].rstrip('?'), "No answer found.")} for question in questions]

    # Write the QA pairs to a JSON file
    with open('qa_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
