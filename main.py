import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import random
import time

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize client
client = InferenceClient(token=HF_TOKEN)

# Available models
MODELS = {
    "Zephyr 7B": "HuggingFaceH4/zephyr-7b-beta",
    "Phi-3": "microsoft/Phi-3-mini-4k-instruct",
}

# Affirmations
AFFIRMATIONS = [
    "You are stronger than you think. 💪",
    "Every step forward is progress. 🪴",
    "You deserve peace and happiness. 😊",
    "It’s okay to feel what you're feeling. 💙",
    "You are doing your best, and that’s enough. 🌟",
]

# UI Config
st.set_page_config(
    page_title="MindMate – Mental Wellness Assistant",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    * { font-family: 'Segoe UI', sans-serif; }
    .block-container { padding: 2rem 3rem; }
    div[data-testid="stSidebar"] { background-color: #f0f4f8; }
    .message {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        line-height: 1.6;
        font-size: 1.02rem;
    }
    .user { background-color: #cde0ff; border-left: 5px solid #2a7de1; color: #000; }
    .bot { background-color: #d5f6eb; border-left: 5px solid #1aaf5d; color: #000; }
    .affirmation-box {
        background-color: #fef9e7;
        border-left: 5px solid #ffd700;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 10px;
        font-size: 1.15rem;
        color: #333333;
    }
    @media (prefers-color-scheme: dark) {
        .user { background-color: #294866; color: #fff; }
        .bot { background-color: #1b4c3a; color: #fff; }
        .affirmation-box { background-color: #4c4400; color: #fff8dc; border-left: 5px solid #ffcc00; }
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧘 MindMate – Mental Wellness Assistant")

# Sidebar
with st.sidebar:
    st.header("🧘 MindMate Controls")
    selected_model_name = st.selectbox("Choose a model:", list(MODELS.keys()), index=0)
    selected_model = MODELS[selected_model_name]
    st.markdown("---")
    mood = st.selectbox("How are you feeling today?", ["😊 Happy", "😐 Okay", "😞 Sad", "😠 Angry", "😰 Anxious"])
    st.markdown("---")
    st.subheader("Session")
    clear_chat = st.button("🗑️ Clear Conversation")

# Session init
if "messages" not in st.session_state or clear_chat:
    st.session_state.messages = []

# Display function
def format_message(sender, message, is_user=False):
    css_class = "user" if is_user else "bot"
    icon = "🧑‍💬" if is_user else "🧘"
    name = "You" if is_user else "MindMate"
    return f"""
    <div class='message {css_class}'>
        <strong>{icon} {name}</strong><br>
        {message}
    </div>
    """

# Generate response
def generate_response(user_input):
    system_prompt = (
        f"You are MindMate, a calm, compassionate, and supportive mental wellness assistant. "
        f"User is feeling {mood}. Respond appropriately. "
        f"Listen empathetically, share mindfulness practices and motivational support. "
        f"Always remind users that you are not a licensed therapist.\n\n"
    )
    prompt = f"<|user|>\n{system_prompt}{user_input}\n<|assistant|>"
    response = client.text_generation(
        prompt=prompt,
        model=selected_model,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
        stop_sequences=["<|user|>"]
    )
    return response.strip()

# Show affirmation
st.markdown(f"<div class='affirmation-box'>🌞 {random.choice(AFFIRMATIONS)}</div>", unsafe_allow_html=True)

# Display messages
with st.container():
    for msg in st.session_state.messages:
        st.markdown(format_message(msg["role"], msg["content"], msg["role"] == "user"), unsafe_allow_html=True)

# Suggested prompt buttons
st.markdown("**💬 Try a suggestion:**")
prompts = [
    "I'm feeling anxious lately.",
    "Can you guide me through a short breathing exercise?",
    "I can't focus on studies, what can I do?",
    "Suggest a positive habit to build.",
    "How do I deal with emotional burnout?",
    "I'm feeling lonely – any advice?"
]

cols = st.columns(3)
for i, prompt_text in enumerate(prompts):
    if cols[i % 3].button(prompt_text, key=f"btn_{i}"):
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.spinner("MindMate is listening..."):
            reply = generate_response(prompt_text)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# Chat input
user_input = st.chat_input("You can talk to me about anything...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("MindMate is listening..."):
        reply = generate_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

# About section
with st.expander("ℹ️ About MindMate"):
    st.markdown(f"""
    **MindMate** is your supportive AI companion for mental wellness:

    - 🧘 Calm, empathetic conversation for emotional support  
    - 🌿 Offers mindfulness and grounding exercises  
    - ❌ Does not replace professional therapy  
    - 🌐 Powered by Hugging Face LLMs like **{selected_model_name}**

    Made with ❤️ for Swecha SOAI 2025 by [Vineesh Chowdary Achanta](https://www.linkedin.com/in/vineesh-chowdary-achanta-803bb925b/)
    """)
