import json

def split_and_convert_to_jsonl(json_file_path, jsonl_file_path):
    # Read the entire JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Remove the leading '[' and trailing ']' to facilitate splitting
    content = content.strip()[1:-1]

    # Split the content based on the pattern indicating a new conversation
    conversations = content.split('},\n{')

    # Open the JSONL file for writing
    with open(jsonl_file_path, 'w', encoding='utf-8') as outfile:
        for conversation in conversations:
            # Add missing braces due to split
            conversation = '{' + conversation + '}'
            try:
                # Attempt to parse each split section as an individual JSON object
                json_obj = json.loads(conversation)
                # If successful, write the conversation as a single line in the JSONL file
                json.dump(json_obj, outfile)
                outfile.write('\n')
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")

if __name__ == "__main__":
    split_and_convert_to_jsonl('qa_pairs.json', 'conversations.jsonl')
