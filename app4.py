import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import shelve
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch
from datasets import load_dataset
from trl import SFTTrainer
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset
import re
import huggingface_hub
import streamlit as st
import dotenv
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

st.title("Ich bin Dein physikalisch-chemischer Assistent, stelle mir Eine Frage!")

# Load local model and tokenizer
base_model = "mistralai/Mistral-7B-Instruct-v0.2"
new_model = "serlikopar/Enlighten_Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.bos_token, tokenizer.eos_token

base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    return_dict=True,
    low_cpu_mem_usage=True,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model_reload, new_model)
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Sidebar with a button to delete chat history
with st.sidebar:
    if st.button("Los geht's von vorne!"):
        st.session_state.messages = []
        save_chat_history([])

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.chat_input("Wie kann ich Dir helfen? Stell mir eine Frage!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        response = pipe(f"<s>[INST] {prompt} [/INST]")
        full_response = response[0]['generated_text']
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Save chat history after each interaction
save_chat_history(st.session_state.messages)
