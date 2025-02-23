import os
import torch
import streamlit as st
import joblib
import time

# Import your custom model and tokenizer
from Naive_gpt.model import GPTLanguageModel
from Naive_gpt.config import *
from Tokenizer import LlamaTokenizer, GPT4Tokenizer
from Tokenizer import train_tokenizer, load_tokenizer

# Styling and Configuration
st.set_page_config(
    page_title="Alif LLM الف",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced aesthetics
st.markdown("""
<style>
    .big-font {
        font-size:20px !important;
        font-family: 'Serif', sans-serif;
    }
    .stTextInput > div > div > input {
        background-color: #F6F6F3;
        color: #335095;
    }
    .stButton > button {
        background-color: #BEB09E;
        color: #335095;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #D6D6D6;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# Model and Tokenizer Configuration
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
    # You can add more models here
    # "Model Name": {
    #     "path": "path_to_model.pth",
    #     "tokenizer": "path_to_tokenizer.model",
    #     "vocab_size": xxx,
    #     "tokenizer_type": TokenizerClass
    # }
}

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer based on selected model name"""
    config = MODEL_CONFIGS[model_name]
    
    # Load Tokenizer
    tokenizer = load_tokenizer(config['tokenizer_type'], config['tokenizer'])
    
    # Load Model
    model = GPTLanguageModel(vocab_size=config['vocab_size'])
    model.load(config['path'])
    model.eval()
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_tokens):
    """Generate text using the selected model"""
    # Encode the input prompt
    encoded = tokenizer.encode(prompt)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    
    # Generate text
    out = model.generate(idx=encoded_tensor, max_new_tokens=max_tokens)
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    
    return decoded_text

def main():
    st.title("Alif LLM الف: Generative Language Model")
    
    # Sidebar for Model and Generation Settings
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        # Model Selection
        selected_model = st.selectbox(
            "Choose a Model", 
            list(MODEL_CONFIGS.keys()),
            help="Select the pre-trained language model you want to use."
        )
        
        # Token Generation Limit
        max_tokens = st.slider(
            "Maximum Generation Tokens", 
            min_value=64, 
            max_value=384, 
            value=128,
            help="Control the length of generated text. Higher values mean longer outputs."
        )
        
        # About Section
        st.markdown("### About Alif LLM")
        st.markdown("""
        A custom generative language model trained on Urdu text.
        
        **Features:**
        - Multiple pre-trained models
        - Configurable text generation
        - Urdu language support
        """)
    
    # Main Content Area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Prompt Input
        prompt = st.text_area(
            "Enter your prompt", 
            placeholder="Type an Urdu sentence to complete...",
            height=200,
            help="Enter an incomplete Urdu sentence for the model to complete."
        )
    
    with col2:
        # Generate Button
        generate_button = st.button("Generate Text", type="primary")
    
    # Text Generation
    if generate_button and prompt:
        with st.spinner('Generating text...'):
            try:
                # Load Model and Tokenizer
                model, tokenizer = load_model_and_tokenizer(selected_model)
                
                # Generate Text
                generated_text = generate_text(model, tokenizer, prompt, max_tokens)
                
                # Display Generated Text
                st.subheader("Generated Text")
                st.write(generated_text)
                
                # Optional: Copy to Clipboard
                st.button("📋 Copy to Clipboard", 
                          on_click=lambda: st.write(generated_text), 
                          key="copy_button")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
    
    elif generate_button and not prompt:
        st.warning("Please enter a prompt to generate text.")

if __name__ == "__main__":
    main()

# import time, os, joblib, streamlit as st
# import google.generativeai as genai
# from dotenv import load_dotenv
# load_dotenv()
# GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)

# new_chat_id = f'{time.time()}'
# MODEL_ROLE = 'ai'
# AI_AVATAR_ICON = '✨'

# # Create a data/ folder if it doesn't already exist
# try:
#     os.mkdir('data/')
# except:
#     # data/ folder already exists
#     pass

# # Load past chats (if available)
# try:
#     past_chats: dict = joblib.load('data/past_chats_list')
# except:
#     past_chats = {}

# # Sidebar allows a list of past chats
# with st.sidebar:
#     st.write('# Past Chats')
#     if st.session_state.get('chat_id') is None:
#         st.session_state.chat_id = st.selectbox(
#             label='Pick a past chat',
#             options=[new_chat_id] + list(past_chats.keys()),
#             format_func=lambda x: past_chats.get(x, 'New Chat'),
#             placeholder='_',
#         )
#     else:
#         # This will happen the first time AI response comes in
#         st.session_state.chat_id = st.selectbox(
#             label='Pick a past chat',
#             options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
#             index=1,
#             format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
#             placeholder='_',
#         )
#     # Save new chats after a message has been sent to AI
#     # TODO: Give user a chance to name chat
#     st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

# st.write('# Chat with Gemini')

# # Chat history (allows to ask multiple questions)
# try:
#     st.session_state.messages = joblib.load(
#         f'data/{st.session_state.chat_id}-st_messages'
#     )
#     st.session_state.gemini_history = joblib.load(
#         f'data/{st.session_state.chat_id}-gemini_messages'
#     )
#     print('old cache')
# except:
#     st.session_state.messages = []
#     st.session_state.gemini_history = []
#     print('new_cache made')
# st.session_state.model = genai.GenerativeModel('gemini-pro')
# st.session_state.chat = st.session_state.model.start_chat(
#     history=st.session_state.gemini_history,
# )

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(
#         name=message['role'],
#         avatar=message.get('avatar'),
#     ):
#         st.markdown(message['content'])

# # React to user input
# if prompt := st.chat_input('Your message here...'):
#     # Save this as a chat for later
#     if st.session_state.chat_id not in past_chats.keys():
#         past_chats[st.session_state.chat_id] = st.session_state.chat_title
#         joblib.dump(past_chats, 'data/past_chats_list')
#     # Display user message in chat message container
#     with st.chat_message('user'):
#         st.markdown(prompt)
#     # Add user message to chat history
#     st.session_state.messages.append(
#         dict(
#             role='user',
#             content=prompt,
#         )
#     )
#     ## Send message to AI
#     response = st.session_state.chat.send_message(
#         prompt,
#         stream=True,
#     )
#     # Display assistant response in chat message container
#     with st.chat_message(
#         name=MODEL_ROLE,
#         avatar=AI_AVATAR_ICON,
#     ):
#         message_placeholder = st.empty()
#         full_response = ''
#         assistant_response = response
#         # Streams in a chunk at a time
#         for chunk in response:
#             # Simulate stream of chunk
#             # TODO: Chunk missing `text` if API stops mid-stream ("safety"?)
#             for ch in chunk.text.split(' '):
#                 full_response += ch + ' '
#                 time.sleep(0.05)
#                 # Rewrites with a cursor at end
#                 message_placeholder.write(full_response + '▌')
#         # Write full message with placeholder
#         message_placeholder.write(full_response)

#     # Add assistant response to chat history
#     st.session_state.messages.append(
#         dict(
#             role=MODEL_ROLE,
#             content=st.session_state.chat.history[-1].parts[0].text,
#             avatar=AI_AVATAR_ICON,
#         )
#     )
#     st.session_state.gemini_history = st.session_state.chat.history
#     # Save to file
#     joblib.dump(
#         st.session_state.messages,
#         f'data/{st.session_state.chat_id}-st_messages',
#     )
#     joblib.dump(
#         st.session_state.gemini_history,
#         f'data/{st.session_state.chat_id}-gemini_messages',
#     )


### Config.toml
# [theme]
# primaryColor="#BEB09E"
# backgroundColor="#F6F6F3"
# secondaryBackgroundColor="#D6D6D6"
# textColor="#335095"
# font="serif"
# # 335095
