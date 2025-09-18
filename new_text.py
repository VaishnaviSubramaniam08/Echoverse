import streamlit as st
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS
from io import BytesIO
import base64
from datetime import datetime
import nltk
from langdetect import detect
from google_trans_new import google_translator as Translator

# Check for required dependencies
try:
    import google_trans_new
    import gtts
    import transformers
except ImportError as e:
    st.error(f"Missing critical dependency: {str(e)}. Please run `pip install google-trans-new gtts transformers langdetect nltk`")
    st.stop()

# Download NLTK data for sentence tokenization
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {str(e)}")

# Page config
st.set_page_config(
    page_title="EchoVerse Studio - Multilingual",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (unchanged from your original code)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
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
        --gradient-multilingual: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        --shadow-primary: 0 10px 30px rgba(79, 70, 229, 0.3);
        --shadow-card: 0 8px 25px rgba(0, 0, 0, 0.4);
    }
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    .stApp {
        background: var(--primary-bg);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
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
        background: var(--gradient-multilingual);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
    }
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
    .language-card {
        background: var(--accent-bg);
        border: 2px solid var(--border-color);
        border-radius: 0.75rem;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    .detected-language {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--success-color);
        margin-bottom: 0.5rem;
    }
    .language-confidence {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    .translation-preview {
        background: var(--accent-bg);
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        min-height: 100px;
        line-height: 1.8;
        white-space: pre-wrap;
        color: var(--text-primary);
        border: 2px solid var(--success-color);
    }
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
    .stFileUploader > div {
        background: var(--accent-bg);
        border: 2px dashed var(--border-color);
        border-radius: 1rem;
    }
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
    .generate-button {
        background: var(--gradient-multilingual);
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
        box-shadow: 0 15px 35px rgba(67, 233, 123, 0.4);
    }
    .audio-section {
        background: var(--accent-bg);
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
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
    .stSuccess, .stInfo {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid var(--success-color);
        color: var(--success-color);
    }
    .stSpinner {
        color: var(--accent-color);
    }
    .language-flag {
        font-size: 1.2em;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Language configurations
SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'flag': '🇺🇸', 'voices': ['us', 'co.uk', 'com.au']},
    'es': {'name': 'Spanish', 'flag': '🇪🇸', 'voices': ['es', 'com.mx']},
    'fr': {'name': 'French', 'flag': '🇫🇷', 'voices': ['fr', 'ca']},
    'de': {'name': 'German', 'flag': '🇩🇪', 'voices': ['de']},
    'it': {'name': 'Italian', 'flag': '🇮🇹', 'voices': ['it']},
    'pt': {'name': 'Portuguese', 'flag': '🇧🇷', 'voices': ['com.br', 'pt']},
    'ru': {'name': 'Russian', 'flag': '🇷🇺', 'voices': ['ru']},
    'ja': {'name': 'Japanese', 'flag': '🇯🇵', 'voices': ['co.jp']},
    'ko': {'name': 'Korean', 'flag': '🇰🇷', 'voices': ['co.kr']},
    'zh': {'name': 'Chinese', 'flag': '🇨🇳', 'voices': ['cn', 'com.tw']},
    'hi': {'name': 'Hindi', 'flag': '🇮🇳', 'voices': ['co.in']},
    'ar': {'name': 'Arabic', 'flag': '🇸🇦', 'voices': ['ae']},
    'nl': {'name': 'Dutch', 'flag': '🇳🇱', 'voices': ['nl']},
    'sv': {'name': 'Swedish', 'flag': '🇸🇪', 'voices': ['se']},
    'no': {'name': 'Norwegian', 'flag': '🇳🇴', 'voices': ['no']},
    'da': {'name': 'Danish', 'flag': '🇩🇰', 'voices': ['dk']},
    'fi': {'name': 'Finnish', 'flag': '🇫🇮', 'voices': ['fi']},
    'pl': {'name': 'Polish', 'flag': '🇵🇱', 'voices': ['pl']},
    'tr': {'name': 'Turkish', 'flag': '🇹🇷', 'voices': ['com.tr']},
    'th': {'name': 'Thai', 'flag': '🇹🇭', 'voices': ['co.th']},
    'vi': {'name': 'Vietnamese', 'flag': '🇻🇳', 'voices': ['vn']},
}

VOICE_STYLES = {
    'calm': {'name': 'Calm & Soothing', 'icon': '😌'},
    'energetic': {'name': 'Energetic & Dynamic', 'icon': '⚡'},
    'professional': {'name': 'Professional & Clear', 'icon': '💼'},
    'storytelling': {'name': 'Storytelling & Dramatic', 'icon': '📚'},
    'friendly': {'name': 'Friendly & Warm', 'icon': '😊'}
}

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'original_text' not in st.session_state:
    st.session_state.original_text = ""
if 'rewritten_text' not in st.session_state:
    st.session_state.rewritten_text = ""
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""
if 'detected_language' not in st.session_state:
    st.session_state.detected_language = None
if 'translation_needed' not in st.session_state:
    st.session_state.translation_needed = False
if 'speech_speed' not in st.session_state:
    st.session_state.speech_speed = 1.0
if 'pause_duration' not in st.session_state:
    st.session_state.pause_duration = 0.5

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
        st.error(f"Error loading model: {str(e)}. Using fallback text.")
        return None, None, None

# Initialize translator
@st.cache_resource
def load_translator():
    try:
        return Translator()
    except Exception as e:
        st.error(f"Error initializing translator: {str(e)}")
        return None

# Function to detect language
def detect_language(text):
    try:
        if not text.strip():
            return 'en', 0.5
        detected_lang = detect(text)
        confidence = 0.95  # Simulated confidence score
        return detected_lang, confidence
    except Exception as e:
        st.error(f"Error detecting language: {str(e)}")
        return 'en', 0.5

# Function to translate text
def translate_text(text, source_lang, target_lang):
    if source_lang == target_lang or not text.strip():
        return text
    try:
        translator = load_translator()
        if translator is None:
            return text
        result = translator.translate(text, lang_src=source_lang, lang_tgt=target_lang)
        return result
    except Exception as e:
        st.error(f"Error translating text: {str(e)}. Using original text.")
        return text

# Function to rewrite text with a specific tone
def rewrite_text(text, tone, language='en'):
    pipe, tokenizer, model = load_model()
    if pipe is None:
        return f"[{tone.upper()} VERSION]\n\n{text}\n\n(Note: Model not available for rewriting)"
    
    try:
        lang_name = SUPPORTED_LANGUAGES.get(language, {}).get('name', 'English')
        prompt = f"""
        You are an expert writer tasked with rewriting the following text in a {tone} tone while preserving its original meaning and enhancing its expressiveness. The text should be optimized for {lang_name} narration and audiobook production. Ensure the rewritten text is concise, natural, and emotionally engaging. Here is the text:

        {text}

        Provide only the rewritten text as the output, maintaining the same language ({lang_name}).
        """
        messages = [{"role": "user", "content": prompt}]
        response = pipe(messages, max_new_tokens=500, num_return_sequences=1)
        rewritten_text = response[0]["generated_text"][-1]["content"]
        return rewritten_text.strip()
    except Exception as e:
        st.error(f"Error rewriting text: {str(e)}")
        return text

# Function to convert text to speech with multilingual support
@st.cache_data
def text_to_speech_multilingual(text, language='en', voice_domain='us', voice_style='calm', speech_speed=1.0, pause_duration=0.5):
    try:
        # Adjust speech parameters based on voice style
        slow_param = voice_style == 'calm'
        
        # Split text into sentences for chunked processing
        sentences = nltk.sent_tokenize(text)
        audio_segments = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            # Generate audio for each sentence
            tts = gTTS(text=sentence, lang=language, tld=voice_domain, slow=slow_param)
            audio_fp = BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            audio_segments.append(audio_fp.read())
            # Add pause (simulated by empty audio or silence could be added in post-processing)
            if pause_duration > 0:
                silence = BytesIO()
                silence.write(b'\x00' * int(44100 * pause_duration))  # 44.1kHz sample rate
                silence.seek(0)
                audio_segments.append(silence.read())
        
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
    try:
        audio_fp.seek(0)
        audio_bytes = audio_fp.read()
        b64 = base64.b64encode(audio_bytes).decode()
        return f"data:audio/mp3;base64,{b64}"
    except Exception as e:
        st.error(f"Error converting audio to base64: {str(e)}")
        return None

# Custom header
st.markdown("""
<div class="custom-header">
    <div class="logo-container">
        🌍 EchoVerse Studio - Multilingual
    </div>
</div>
""", unsafe_allow_html=True)

# Main layout - Two columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
    <div class="zone-container">
        <h2 class="zone-title">📝 Text Input & Language Detection</h2>
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
            height=150,
            placeholder="Paste your text in any language to transform it into an expressive multilingual audiobook...",
            key="text_input"
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a .txt file",
            type="txt",
            help="Drag & drop your .txt file here or click to browse"
        )
        if uploaded_file:
            try:
                input_text = uploaded_file.read().decode("utf-8")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Store original text and detect language
    if input_text and input_text != st.session_state.original_text:
        st.session_state.original_text = input_text
        detected_lang, confidence = detect_language(input_text)
        st.session_state.detected_language = detected_lang
        
        # Display detected language
        if detected_lang in SUPPORTED_LANGUAGES:
            lang_info = SUPPORTED_LANGUAGES[detected_lang]
            st.markdown(f"""
            <div class="language-card">
                <div class="detected-language">
                    <span class="language-flag">{lang_info['flag']}</span>
                    Detected Language: {lang_info['name']}
                </div>
                <div class="language-confidence">Confidence: {confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"Detected language '{detected_lang}' is not fully supported. Defaulting to English.")
    
    # Language and translation settings
    st.markdown("### 🌐 Language & Translation")
    
    # Target language selection
    lang_options = [(code, f"{info['flag']} {info['name']}") for code, info in SUPPORTED_LANGUAGES.items()]
    target_language = st.selectbox(
        "Target Language for Audio:",
        options=lang_options,
        format_func=lambda x: x[1],
        key="target_language"
    )[0]
    
    # Check if translation is needed
    if st.session_state.detected_language and st.session_state.detected_language != target_language:
        st.session_state.translation_needed = True
        st.info(f"Translation will be performed: {SUPPORTED_LANGUAGES.get(st.session_state.detected_language, {}).get('name', 'Unknown')} → {SUPPORTED_LANGUAGES[target_language]['name']}")
    else:
        st.session_state.translation_needed = False
    
    # Voice configuration
    st.markdown("### 🎭 Voice Configuration")
    
    # Voice style selection
    style_col1, style_col2 = st.columns(2)
    
    with style_col1:
        voice_style = st.selectbox(
            "Voice Style:",
            options=list(VOICE_STYLES.keys()),
            format_func=lambda x: f"{VOICE_STYLES[x]['icon']} {VOICE_STYLES[x]['name']}",
            key="voice_style"
        )
    
    with style_col2:
        # Available voice domains for target language
        available_voices = SUPPORTED_LANGUAGES[target_language]['voices']
        voice_domain = st.selectbox(
            "Voice Region:",
            options=available_voices,
            key="voice_domain"
        )
    
    # Tone selection
    st.markdown("### 🎨 Tone Selection")
    
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
    
    st.info(f"Selected: **{selected_tone}** tone with **{VOICE_STYLES[voice_style]['name']}** style")
    
    # Generate button
    st.markdown("---")
    generate_clicked = st.button(
        "🎨 Generate Multilingual Audiobook",
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
    preview_options = ["Original"]
    if st.session_state.translation_needed:
        preview_options.append("Translated")
    preview_options.append("Tone-Adapted")
    
    preview_tab = st.radio(
        "Preview:",
        preview_options,
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
    elif preview_tab == "Translated":
        if st.session_state.translated_text:
            st.markdown(f"""
            <div class="translation-preview">
                {st.session_state.translated_text}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="text-preview empty">
                <div>Translated text will appear here...</div>
            </div>
            """, unsafe_allow_html=True)
    else:  # Tone-Adapted
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
    st.markdown("### 🎵 Multilingual Audio Output")
    
    if 'audio_data' in st.session_state and st.session_state.audio_data:
        st.audio(st.session_state.audio_data, format="audio/mp3")
        
        # Audio info
        if 'last_generation_info' in st.session_state:
            info = st.session_state.last_generation_info
            st.caption(f"🌍 Language: {info['language']} | 🎭 Style: {info['style']} | 🎨 Tone: {info['tone']}")
        
        # Download button
        filename = f"echoverse_{target_language}_{selected_tone.lower()}_{voice_style}_audio.mp3"
        if st.download_button(
            label="💾 Download Multilingual MP3",
            data=st.session_state.audio_data,
            file_name=filename,
            mime="audio/mp3",
            key="download_audio"
        ):
            st.success("Multilingual audio downloaded successfully!")
    else:
        st.info("Multilingual audio will appear here after generation...")

# Process generation
if generate_clicked:
    if not st.session_state.original_text.strip():
        st.error("Please provide text input.")
    else:
        with st.spinner("🌍 Processing multilingual audiobook..."):
            progress_bar = st.progress(0)
            
            # Step 1: Translation (if needed)
            progress_bar.progress(20)
            if st.session_state.translation_needed:
                st.info("🔄 Translating text...")
                translated_text = translate_text(
                    st.session_state.original_text, 
                    st.session_state.detected_language, 
                    target_language
                )
                st.session_state.translated_text = translated_text
                working_text = translated_text
            else:
                working_text = st.session_state.original_text
                st.session_state.translated_text = working_text
            
            # Step 2: Tone adaptation
            progress_bar.progress(40)
            st.info("🎨 Adapting tone...")
            rewritten_text = rewrite_text(working_text, selected_tone, target_language)
            st.session_state.rewritten_text = rewritten_text
            
            # Step 3: Audio generation
            progress_bar.progress(60)
            st.info("🎵 Generating multilingual audio...")
            audio_fp = text_to_speech_multilingual(
                st.session_state.rewritten_text, 
                target_language, 
                voice_domain, 
                voice_style,
                speech_speed=st.session_state.speech_speed,
                pause_duration=st.session_state.pause_duration
            )
            
            # Step 4: Finalization
            progress_bar.progress(80)
            if audio_fp:
                audio_fp.seek(0)
                st.session_state.audio_data = audio_fp.read()
                
                # Store generation info
                st.session_state.last_generation_info = {
                    'language': SUPPORTED_LANGUAGES[target_language]['name'],
                    'style': VOICE_STYLES[voice_style]['name'],
                    'tone': selected_tone,
                    'voice_domain': voice_domain
                }
            
            progress_bar.progress(100)
            
            # Add to history
            history_item = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_language": SUPPORTED_LANGUAGES.get(st.session_state.detected_language, {}).get('name', 'Unknown'),
                "target_language": SUPPORTED_LANGUAGES[target_language]['name'],
                "tone": selected_tone,
                "voice_style": VOICE_STYLES[voice_style]['name'],
                "voice_domain": voice_domain,
                "translation_needed": st.session_state.translation_needed,
                "preview": st.session_state.original_text[:50] + "..." if len(st.session_state.original_text) > 50 else st.session_state.original_text,
                "original_text": st.session_state.original_text,
                "translated_text": st.session_state.translated_text,
                "rewritten_text": st.session_state.rewritten_text
            }
            st.session_state.history.insert(0, history_item)
            
            # Keep only last 5 items
            if len(st.session_state.history) > 5:
                st.session_state.history = st.session_state.history[:5]
            
            progress_bar.empty()
            st.success("🎉 Multilingual audiobook generated successfully!")
            
            # Show generation summary
            col_sum1, col_sum2 = st.columns(2)
            with col_sum1:
                st.metric("Source Language", 
                         SUPPORTED_LANGUAGES.get(st.session_state.detected_language, {}).get('name', 'Unknown'))
                st.metric("Voice Style", VOICE_STYLES[voice_style]['name'])
            with col_sum2:
                st.metric("Target Language", SUPPORTED_LANGUAGES[target_language]['name'])
                st.metric("Tone Applied", selected_tone)
            
            st.balloons()

# Advanced Features Section
st.markdown("---")
with st.expander("🔧 Advanced Features"):
    st.markdown("### 📊 Text Analysis")
    
    if st.session_state.original_text:
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
        
        with analysis_col1:
            word_count = len(st.session_state.original_text.split())
            st.metric("Word Count", word_count)
        
        with analysis_col2:
            char_count = len(st.session_state.original_text)
            st.metric("Character Count", char_count)
        
        with analysis_col3:
            reading_time = max(1, round(word_count / 200))  # 200 words per minute
            st.metric("Est. Reading Time", f"{reading_time} min")
    
    st.markdown("### 🎚️ Voice Customization")
    
    # Additional voice parameters
    voice_col1, voice_col2 = st.columns(2)
    
    with voice_col1:
        st.session_state.speech_speed = st.slider(
            "Speech Speed",
            min_value=0.5,
            max_value=2.0,
            value=st.session_state.speech_speed,
            step=0.1,
            help="Adjust the speaking rate (0.5 = slow, 2.0 = fast)"
        )
    
    with voice_col2:
        st.session_state.pause_duration = st.slider(
            "Pause Between Sentences",
            min_value=0.1,
            max_value=2.0,
            value=st.session_state.pause_duration,
            step=0.1,
            help="Duration of pauses between sentences in seconds"
        )
    
    st.markdown("### 📈 Language Statistics")
    
    if st.session_state.history:
        lang_usage = {}
        tone_usage = {}
        
        for item in st.session_state.history:
            lang = item.get('target_language', 'Unknown')
            tone = item.get('tone', 'Unknown')
            lang_usage[lang] = lang_usage.get(lang, 0) + 1
            tone_usage[tone] = tone_usage.get(tone, 0) + 1
        
        stat_col1, stat_col2 = st.columns(2)
        
        with stat_col1:
            st.write("**Most Used Languages:**")
            for lang, count in sorted(lang_usage.items(), key=lambda x: x[1], reverse=True):
                flag = next((info['flag'] for info in SUPPORTED_LANGUAGES.values() if info['name'] == lang), '🌐')
                st.write(f"{flag} {lang}: {count}")
        
        with stat_col2:
            st.write("**Most Used Tones:**")
            for tone, count in sorted(tone_usage.items(), key=lambda x: x[1], reverse=True):
                icon = {'Neutral': '📝', 'Suspenseful': '🔥', 'Inspiring': '🌟'}.get(tone, '🎭')
                st.write(f"{icon} {tone}: {count}")

# Batch Processing Section
st.markdown("---")
with st.expander("📦 Batch Processing"):
    st.markdown("### Upload Multiple Files")
    
    batch_files = st.file_uploader(
        "Upload multiple .txt files for batch processing",
        type="txt",
        accept_multiple_files=True,
        help="Select multiple text files to process them all at once"
    )
    
    if batch_files:
        st.write(f"Selected {len(batch_files)} files:")
        for file in batch_files:
            st.write(f"• {file.name}")
        
        batch_col1, batch_col2 = st.columns(2)
        
        with batch_col1:
            batch_target_lang = st.selectbox(
                "Batch Target Language:",
                options=lang_options,
                format_func=lambda x: x[1],
                key="batch_target_language"
            )[0]
        
        with batch_col2:
            batch_tone = st.selectbox(
                "Batch Tone:",
                ["Neutral", "Suspenseful", "Inspiring"],
                key="batch_tone"
            )
        
        if st.button("🚀 Process All Files", key="batch_process"):
            if not batch_files:
                st.error("No files selected for batch processing.")
            else:
                batch_results = []
                batch_progress = st.progress(0)
                
                for i, file in enumerate(batch_files):
                    try:
                        # Read file content
                        content = file.read().decode("utf-8")
                        if not content.strip():
                            st.warning(f"Skipping empty file: {file.name}")
                            continue
                        
                        # Detect language
                        detected_lang, _ = detect_language(content)
                        
                        # Translate if needed
                        if detected_lang != batch_target_lang:
                            translated = translate_text(content, detected_lang, batch_target_lang)
                        else:
                            translated = content
                        
                        # Rewrite with tone
                        rewritten = rewrite_text(translated, batch_tone, batch_target_lang)
                        
                        # Generate audio
                        audio_fp = text_to_speech_multilingual(
                            rewritten, 
                            batch_target_lang, 
                            SUPPORTED_LANGUAGES[batch_target_lang]['voices'][0], 
                            'professional',
                            speech_speed=st.session_state.speech_speed,
                            pause_duration=st.session_state.pause_duration
                        )
                        
                        if audio_fp:
                            audio_fp.seek(0)
                            audio_data = audio_fp.read()
                            
                            batch_results.append({
                                'filename': file.name,
                                'audio_data': audio_data,
                                'processed_text': rewritten
                            })
                        
                        batch_progress.progress((i + 1) / len(batch_files))
                        
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                
                # Display results
                if batch_results:
                    st.success(f"Successfully processed {len(batch_results)} files!")
                    
                    for result in batch_results:
                        with st.expander(f"📄 {result['filename']}"):
                            st.audio(result['audio_data'], format="audio/mp3")
                            st.download_button(
                                label=f"💾 Download {result['filename'].replace('.txt', '.mp3')}",
                                data=result['audio_data'],
                                file_name=f"{result['filename'].replace('.txt', '')}_audio.mp3",
                                mime="audio/mp3",
                                key=f"download_batch_{result['filename']}"
                            )

# History section with enhanced multilingual features
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📚 Multilingual Project History")
    
    # Clear history button
    if st.button("🗑️ Clear History", key="clear_history"):
        st.session_state.history = []
        st.rerun()
    
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        language_filter = st.selectbox(
            "Filter by Language:",
            ["All Languages"] + list(set([item['target_language'] for item in st.session_state.history])),
            key="language_filter"
        )
    
    with filter_col2:
        tone_filter = st.selectbox(
            "Filter by Tone:",
            ["All Tones"] + list(set([item['tone'] for item in st.session_state.history])),
            key="tone_filter"
        )
    
    # Apply filters
    filtered_history = st.session_state.history
    if language_filter != "All Languages":
        filtered_history = [item for item in filtered_history if item['target_language'] == language_filter]
    if tone_filter != "All Tones":
        filtered_history = [item for item in filtered_history if item['tone'] == tone_filter]
    
    # Display filtered history
    for i, item in enumerate(filtered_history):
        with st.expander(f"{item['target_language']} • {item['tone']} • {item['voice_style']} • {item['timestamp']}"):
            st.write(f"**Preview:** {item['preview']}")
            
            if item['translation_needed']:
                st.info(f"🔄 Translated from {item['source_language']} to {item['target_language']}")
            
            hist_col1, hist_col2, hist_col3 = st.columns(3)
            
            with hist_col1:
                if st.button(f"🔄 Reload Original", key=f"reload_orig_{i}"):
                    st.session_state.original_text = item['original_text']
                    st.rerun()
            
            with hist_col2:
                if st.button(f"🌍 Reload Translated", key=f"reload_trans_{i}"):
                    st.session_state.translated_text = item['translated_text']
                    st.rerun()
            
            with hist_col3:
                if st.button(f"🎨 Reload Rewritten", key=f"reload_rewrite_{i}"):
                    st.session_state.rewritten_text = item['rewritten_text']
                    st.rerun()

# Footer with additional information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-secondary); padding: 2rem;">
    <h4>🌍 EchoVerse Studio - Multilingual Edition</h4>
    <p>Transform text from any language into expressive audiobooks with AI-powered tone adaptation</p>
    <p><strong>Supported Languages:</strong> {}</p>
    <p><strong>Voice Styles:</strong> {}</p>
    <p><em>Built with Streamlit • Powered by AI • Enhanced with gTTS</em></p>
</div>
""".format(
    len(SUPPORTED_LANGUAGES),
    len(VOICE_STYLES)
), unsafe_allow_html=True)