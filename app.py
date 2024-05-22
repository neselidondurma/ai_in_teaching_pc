import os
import streamlit as st
from transformers import AutoTokenizer, pipeline, set_seed
import dotenv

dotenv.load_dotenv("hugging_face.env")
huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
print(huggingface_token)
command = f"huggingface-cli login --token {huggingface_token}"
os.system(command)
base_model = "mistralai/Mistral-7B-Instruct-v0.2"
new_model = "serlikopar/Enlighten_Instruct"

# Load tokenizer
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

logging.set_verbosity(logging.CRITICAL)

def build_prompt(question):
  prompt=f"<s>[INST] {question} [/INST]"
  return prompt

def get_pipeline():
    return pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

pipe = get_pipeline()

st.title('Text Generation Model')
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


