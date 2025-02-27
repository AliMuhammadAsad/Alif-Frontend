import os
import torch
import gradio as gr
import joblib
from Naive_gpt.model import GPTLanguageModel
from Naive_gpt.config import *
from Tokenizer import LlamaTokenizer, GPT4Tokenizer
from Tokenizer import train_tokenizer, load_tokenizer

# Model Configurations
MODEL_CONFIGS = {
    "Naive GPT Weights 0_80 (DGX)": {
        "path": "model_weights_epoch_0_80.pth",
        "tokenizer": "llama_Uwiki_2.model",
        "vocab_size": 10000,
        "tokenizer_type": LlamaTokenizer
    },
    "Naive GPT Weights 0_78 (DGX)": {
        "path": "model_weights_epoch_0_78.pth",
        "tokenizer": "llama_Uwiki_2.model",
        "vocab_size": 10000,
        "tokenizer_type": LlamaTokenizer
    },
    "Naive GPT Weights 0_25 (DGX)": {
        "path": "model_weights_epoch_0_25.pth",
        "tokenizer": "llama_Uwiki_2.model",
        "vocab_size": 10000,
        "tokenizer_type": LlamaTokenizer
    },
    "Naive GPT Weights 0_69 (HPC)": {
        "path": "model_weights_epoch_0_69.pth",
        "tokenizer": "llama_Uwiki_2.model",
        "vocab_size": 10000,
        "tokenizer_type": LlamaTokenizer
    },
}

# Load model and tokenizer
def load_model_and_tokenizer(model_name):
    config = MODEL_CONFIGS[model_name]
    tokenizer = load_tokenizer(config['tokenizer_type'], config['tokenizer'])
    model = GPTLanguageModel(vocab_size=config['vocab_size'])
    model.load(config['path'])
    model.eval()
    return model, tokenizer

# Generate response
def generate_text(history, prompt, model_name, max_tokens):
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    if history:
        conversation = " ".join([entry[0] + " " + entry[1] for entry in history])
        prompt = conversation + " " + prompt  # Append new prompt to history
    
    encoded = tokenizer.encode(prompt)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    
    out = model.generate(idx=encoded_tensor, max_new_tokens=max_tokens)
    
    response = tokenizer.decode(out.squeeze(0).tolist())
    
    history.append((prompt, response))  # Save the new response in history
    return history, ""

# About Section
about_text = """
## About Alif LLM الف  
A custom generative language model trained on Urdu text.  

### Features:  
✅ Multiple pre-trained models  
✅ Configurable text generation  
✅ Urdu language support  
"""

# UI with improved aesthetics
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("<h1 style='text-align: center;'>📖 Alif LLM الف - Urdu Language Model</h1>")
    
    with gr.Row():
        model_selector = gr.Dropdown(choices=list(MODEL_CONFIGS.keys()), label="Choose a Model", value=list(MODEL_CONFIGS.keys())[0])
        max_tokens_slider = gr.Slider(minimum=64, maximum=384, value=128, label="Max Tokens", interactive=True)
    
    chatbot = gr.Chatbot(label="Chat with Alif LLM الف", height=400)
    user_input = gr.Textbox(label="Your Message", placeholder="Type an Urdu sentence...", interactive=True)
    
    with gr.Row():
        send_button = gr.Button("📝 Send", variant="primary")
        clear_button = gr.Button("🔄 Start New Chat", variant="secondary")

    send_button.click(generate_text, 
                      inputs=[chatbot, user_input, model_selector, max_tokens_slider], 
                      outputs=[chatbot, user_input])
    
    clear_button.click(lambda: [], outputs=[chatbot])
    
    gr.Markdown(about_text)

demo.launch(share=True)