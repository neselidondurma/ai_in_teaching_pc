import os
import panel as pn
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_transformers_logging
import logging

# Set up the web interface
pn.extension()

# Load environment variables and configure logging
load_dotenv("hugging_face.env")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
set_transformers_logging(logging.ERROR)

# Define the model paths
base_model = "mistralai/Mistral-7B-Instruct-v0.2"
new_model = "serlikopar/Enlighten_Instruct"

# Load the tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

# Load the new model, assuming it is a fine-tuned version of the base model
model = AutoModelForCausalLM.from_pretrained(
    new_model,
    torch_dtype=torch.bfloat16,
    return_dict=True,
    trust_remote_code=True
)

# Setup a text-generation pipeline
text_gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=200)

# Define the callback for the chat interface
def callback(event):
    user_input = event.obj.value
    if user_input:
        prompt = f"<s>[INST] {user_input} [/INST]"
        response = text_gen_pipe(prompt)[0]['generated_text']
        chat_area.append(pn.pane.Markdown(f"**Mistral:** {response}"))

# Setup Panel chat interface
chat_area = pn.Column(sizing_mode='stretch_width')
input_box = pn.widgets.TextInput(name='Ask Mistral', placeholder='Type your question here...')
input_box.jscallback(args={'chat_area': chat_area}, value="""
    if (cb_obj.value.trim() !== "") {
        chat_area.append(pn.pane.Markdown("**You:** " + cb_obj.value));
        cb_obj.value = ""; // Clear input box after sending
    }
""")
input_box.param.watch(callback, 'value')

# Servable Panel layout
pn.Column("# Mistral Chat Interface", chat_area, input_box, sizing_mode='stretch_width').servable()

