import streamlit as st
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS
from io import BytesIO
import base64
import time
from datetime import datetime
import nltk
import hashlib

# Download NLTK data for sentence tokenization
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {str(e)}")

# Page config
st.set_page_config(
    page_title="EchoVerse Studio",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for the Narration Studio design
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global variables */
    :root {
        --primary-bg: #0f0f23;
        --secondary-bg: #1a1a2e;
        --accent-bg: #16213e;
        --text-primary: #ffffff;
        --text-secondary: #b3b3b3;
        --accent-color: #4f46e5;
        --accent-hover: #6366f1;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --border-color: #2d2d44;
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-accent: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --shadow-primary: 0 10px 30px rgba(79, 70, 229, 0.3);
        --shadow-card: 0 8px 25px rgba(0, 0, 0, 0.4);
    }
    
    /* Hide Streamlit default elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Main app styling */
    .stApp {
        background: var(--primary-bg);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Custom header */
    .custom-header {
        background: var(--secondary-bg);
        padding: 1rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 1px solid var(--border-color);
        box-shadow: var(--shadow-card);
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.5rem;
        font-weight: 700;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
    }
    
    /* Zone containers */
    .zone-container {
        background: var(--secondary-bg);
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: var(--shadow-card);
        border: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }
    
    .zone-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Custom buttons */
    .tone-button {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        margin: 0.25rem;
        background: var(--accent-bg);
        border: 2px solid var(--border-color);
        border-radius: 2rem;
        color: var(--text-primary);
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .tone-button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-card);
        text-decoration: none;
        color: var(--text-primary);
    }
    
    .tone-button.active {
        background: var(--gradient-primary);
        border-color: transparent;
        color: white;
        box-shadow: var(--shadow-primary);
    }
    
    /* Voice cards */
    .voice-card {
        background: var(--accent-bg);
        border: 2px solid var(--border-color);
        border-radius: 0.75rem;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .voice-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-card);
    }
    
    .voice-card.selected {
        border-color: var(--accent-color);
        background: rgba(79, 70, 229, 0.1);
    }
    
    .voice-name {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .voice-accent {
        color: var(--text-secondary);
        font-size: 0.875rem;
    }
    
    /* Text areas and inputs */
    .stTextArea > div > div > textarea {
        background: var(--accent-bg);
        border: 1px solid var(--border-color);
        border-radius: 0.5rem;
        color: var(--text-primary);
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: var(--accent-bg);
        border: 2px dashed var(--border-color);
        border-radius: 1rem;
    }
    
    /* Text preview containers */
    .text-preview {
        background: var(--accent-bg);
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        min-height: 200px;
        line-height: 1.8;
        white-space: pre-wrap;
        color: var(--text-primary);
    }
    
    .text-preview.empty {
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--text-secondary);
        font-style: italic;
    }
    
    /* Custom generate button */
    .generate-button {
        background: var(--gradient-accent);
        border: none;
        border-radius: 0.75rem;
        padding: 1rem 2rem;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-primary);
        width: 100%;
        margin: 1rem 0;
    }
    
    .generate-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(245, 87, 108, 0.4);
    }
    
    /* Audio section */
    .audio-section {
        background: var(--accent-bg);
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Download button */
    .download-button {
        background: var(--success-color);
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 2rem;
        color: white;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    
    .download-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
    }
    
    /* History section */
    .history-container {
        background: var(--secondary-bg);
        border-top: 1px solid var(--border-color);
        padding: 1rem;
        margin: 2rem -1rem -1rem -1rem;
    }
    
    .history-item {
        background: var(--accent-bg);
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border: 1px solid var(--border-color);
        color: var(--text-primary);
    }
    
    .history-meta {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }
    
    /* Streamlit specific overrides */
    .stSelectbox > div > div {
        background: var(--accent-bg);
        border-color: var(--border-color);
    }
    
    .stButton > button {
        background: var(--accent-color);
        color: white;
        border: none;
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: var(--accent-hover);
        transform: translateY(-2px);
    }
    
    /* Success/Info messages */
    .stSuccess, .stInfo {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid var(--success-color);
        color: var(--success-color);
    }
    
    /* Loading spinner */
    .stSpinner {
        color: var(--accent-color);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'original_text' not in st.session_state:
    st.session_state.original_text = ""
if 'rewritten_text' not in st.session_state:
    st.session_state.rewritten_text = ""

# Initialize the model and tokenizer for text generation
@st.cache_resource
def load_model():
    try:
        model_name = "ibm-granite/granite-3.3-2b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
        return pipe, tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Function to rewrite text with a specific tone
def rewrite_text(text, tone):
    pipe, tokenizer, model = load_model()
    if pipe is None:
        return f"[{tone.upper()} VERSION]\n\n{text}\n\n(Note: Model not available for rewriting, showing original with tone marker)"
    
    try:
        prompt = f"""
        You are an expert writer tasked with rewriting the following text in a {tone} tone while preserving its original meaning and enhancing its expressiveness. Ensure the rewritten text is concise, natural, and suitable for narration. Here is the text:

        {text}

        Provide only the rewritten text as the output.
        """
        messages = [{"role": "user", "content": prompt}]
        response = pipe(messages, max_new_tokens=500, num_return_sequences=1)
        rewritten_text = response[0]["generated_text"][-1]["content"]
        return rewritten_text.strip()
    except Exception as e:
        st.error(f"Error rewriting text: {str(e)}")
        return text

# Function to convert text to speech with caching and chunking
@st.cache_data
def text_to_speech(text, voice="us"):
    try:
        # Split text into sentences for chunked processing
        sentences = nltk.sent_tokenize(text)
        audio_segments = []
        
        for sentence in sentences:
            # Generate audio for each sentence
            tts = gTTS(text=sentence, lang="en", tld=voice, slow=False)
            audio_fp = BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            audio_segments.append(audio_fp.read())
        
        # Concatenate audio segments
        combined_audio = BytesIO()
        for segment in audio_segments:
            combined_audio.write(segment)
        combined_audio.seek(0)
        return combined_audio
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

# Function to convert audio to base64 for streaming
def audio_to_base64(audio_fp):
    audio_fp.seek(0)
    audio_bytes = audio_fp.read()
    b64 = base64.b64encode(audio_bytes).decode()
    return f"data:audio/mp3;base64,{b64}"

# Custom header
st.markdown("""
<div class="custom-header">
    <div class="logo-container">
        🎧 EchoVerse Studio
    </div>
</div>
""", unsafe_allow_html=True)

# Main layout - Two columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
    <div class="zone-container">
        <h2 class="zone-title">📝 Text Input</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ("Paste Text", "Upload .txt File"),
        horizontal=True,
        key="input_method"
    )
    
    input_text = ""
    if input_method == "Paste Text":
        input_text = st.text_area(
            "Paste your text here:",
            height=200,
            placeholder="Paste your text here to transform it into an expressive audiobook...",
            key="text_input"
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a .txt file",
            type="txt",
            help="Drag & drop your .txt file here or click to browse"
        )
        if uploaded_file:
            input_text = uploaded_file.read().decode("utf-8")
    
    # Store original text
    if input_text:
        st.session_state.original_text = input_text
    
    st.markdown("### Select Tone")
    
    # Tone selection with custom styling
    tone_col1, tone_col2, tone_col3 = st.columns(3)
    
    with tone_col1:
        neutral_selected = st.button("📝 Neutral", key="neutral", use_container_width=True)
    with tone_col2:
        suspenseful_selected = st.button("🔥 Suspenseful", key="suspenseful", use_container_width=True)
    with tone_col3:
        inspiring_selected = st.button("🌟 Inspiring", key="inspiring", use_container_width=True)
    
    # Determine selected tone
    selected_tone = "Neutral"
    if suspenseful_selected:
        selected_tone = "Suspenseful"
        st.session_state.selected_tone = "Suspenseful"
    elif inspiring_selected:
        selected_tone = "Inspiring"
        st.session_state.selected_tone = "Inspiring"
    elif neutral_selected:
        selected_tone = "Neutral"
        st.session_state.selected_tone = "Neutral"
    elif 'selected_tone' not in st.session_state:
        st.session_state.selected_tone = "Neutral"
    else:
        selected_tone = st.session_state.selected_tone
    
    st.info(f"Selected tone: *{selected_tone}*")
    
    st.markdown("### Choose Voice")
    
    # Voice selection
    voice = st.selectbox(
        "Select Voice:",
        ["Lisa (US - American English)", "Michael (UK - British English)", "Allison (AU - Australian English)"],
        key="voice_selection"
    )
    
    voice_map = {
        "Lisa (US - American English)": "us",
        "Michael (UK - British English)": "co.uk",
        "Allison (AU - Australian English)": "com.au"
    }
    
    # Generate button
    st.markdown("---")
    generate_clicked = st.button(
        "🎨 Generate Audiobook",
        key="generate",
        use_container_width=True,
        type="primary"
    )

with col2:
    st.markdown("""
    <div class="zone-container">
        <h2 class="zone-title">🎧 Preview Studio</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Preview tabs
    preview_tab = st.radio(
        "Preview:",
        ("Original", "Tone-Adapted"),
        horizontal=True,
        key="preview_tab"
    )
    
    # Text preview
    if preview_tab == "Original":
        if st.session_state.original_text:
            st.markdown(f"""
            <div class="text-preview">
                {st.session_state.original_text}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="text-preview empty">
                <div>Your original text will appear here...</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        if st.session_state.rewritten_text:
            st.markdown(f"""
            <div class="text-preview">
                {st.session_state.rewritten_text}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="text-preview empty">
                <div>Tone-adapted text will appear here...</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Audio section
    st.markdown("### 🎵 Audio Output")
    
    if 'audio_data' in st.session_state and st.session_state.audio_data:
        st.audio(st.session_state.audio_data, format="audio/mp3")
        
        # Download button
        if st.download_button(
            label="💾 Download MP3",
            data=st.session_state.audio_data,
            file_name=f"echoverse_{selected_tone.lower()}_audio.mp3",
            mime="audio/mp3",
            key="download_audio"
        ):
            st.success("Audio downloaded successfully!")
    else:
        st.info("Audio will appear here after generation...")

# Process generation
if generate_clicked:
    if not st.session_state.original_text.strip():
        st.error("Please provide text input.")
    else:
        with st.spinner("Generating audiobook..."):
            # Rewrite text
            rewritten_text = rewrite_text(st.session_state.original_text, selected_tone)
            st.session_state.rewritten_text = rewritten_text
            
            # Generate audio
            audio_fp = text_to_speech(st.session_state.rewritten_text, voice_map[voice])
            if audio_fp:
                audio_fp.seek(0)
                st.session_state.audio_data = audio_fp.read()
            
            # Add to history
            history_item = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tone": selected_tone,
                "voice": voice.split(" (")[0],
                "preview": st.session_state.original_text[:50] + "...",
                "original_text": st.session_state.original_text,
                "rewritten_text": rewritten_text
            }
            st.session_state.history.insert(0, history_item)
            
            # Keep only last 5 items
            if len(st.session_state.history) > 5:
                st.session_state.history = st.session_state.history[:5]
            
            st.success("🎉 Audiobook generated successfully!")
            st.balloons()

# History section
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📚 Recent Projects")
    
    for i, item in enumerate(st.session_state.history):
        with st.expander(f"{item['tone']} • {item['voice']} • {item['timestamp']}"):
            st.write(f"*Preview:* {item['preview']}")
            
            hist_col1, hist_col2 = st.columns(2)
            with hist_col1:
                if st.button(f"🔄 Reload Original", key=f"reload_orig_{i}"):
                    st.session_state.original_text = item['original_text']
                    st.rerun()
            
            with hist_col2:
                if st.button(f"🔄 Reload Rewritten", key=f"reload_rewrite_{i}"):
                    st.session_state.rewritten_text = item['rewritten_text']
                    st.rerun()