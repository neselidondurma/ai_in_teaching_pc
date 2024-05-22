import streamlit as st
from transformers import AutoTokenizer, pipeline

# Define a custom hash function for tokenizers
def hash_tokenizer(tokenizer):
    return 1  # Return a constant value for the tokenizer

@st.cache_data(hash_funcs={AutoTokenizer: hash_tokenizer})
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", trust_remote_code=True)
    model = pipeline('text-generation', model="serlikopar/Enlighten_Instruct", tokenizer=tokenizer, max_length=200)
    return model

# Initialize the pipeline once
pipe = load_model()

def build_prompt(question):
  prompt=f"<s>[INST] {question} [/INST]"
  return prompt

# Streamlit application code
st.title('Ich bin Dein physikalisch-chemischer Assistent, stelle mir Eine Frage! ')
question = st.text_input("Enter your question:")

if st.button('Generate Answer'):
    if question:
        with st.spinner('Generating...'):
            prompt = build_prompt(question)
            result = pipe(prompt)
            generated_text = result[0]['generated_text']
        st.write(generated_text)
    else:
        st.write("Please enter a question to generate an answer.")