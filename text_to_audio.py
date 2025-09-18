# EchoVerse Studio - Minimal Dependencies Version
# This version avoids ALL heavy dependencies and TensorFlow issues

import streamlit as st
import os
import re
from io import BytesIO
from datetime import datetime
import string
import random

# Only try to import gTTS - no other heavy libraries
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="EchoVerse Studio - Minimal",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --primary-bg: #0a0a1a;
        --secondary-bg: #1a1a2e;
        --accent-bg: #16213e;
        --text-primary: #ffffff;
        --text-secondary: #b3b3b3;
        --accent-color: #4f46e5;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --border-color: #2d2d44;
    }
    
    .stApp {
        background: var(--primary-bg);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        text-align: center;
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
        font-size: 2rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .status-card {
        background: var(--secondary-bg);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid var(--success-color);
        margin: 0.5rem 0;
    }
    
    .status-card.warning {
        border-left-color: var(--warning-color);
    }
    
    .status-card.error {
        border-left-color: var(--error-color);
    }
    
    .section-container {
        background: var(--secondary-bg);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
    }
    
    .section-title {
        color: var(--text-primary);
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .text-preview-box {
        background: var(--accent-bg);
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        min-height: 120px;
        color: var(--text-primary);
        line-height: 1.6;
        white-space: pre-wrap;
        border: 1px solid var(--border-color);
    }
    
    .empty-state {
        color: var(--text-secondary);
        font-style: italic;
        text-align: center;
        padding: 2rem;
    }
    
    .language-badge {
        display: inline-block;
        background: var(--accent-color);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .tone-button {
        background: var(--accent-bg);
        border: 2px solid var(--border-color);
        color: var(--text-primary);
        padding: 0.75rem 1.5rem;
        border-radius: 2rem;
        font-weight: 500;
        margin: 0.25rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .tone-button:hover {
        border-color: var(--accent-color);
        background: rgba(79, 70, 229, 0.1);
    }
    
    .tone-button.selected {
        background: var(--accent-color);
        border-color: var(--accent-color);
        color: white;
    }
    
    /* Override Streamlit defaults */
    .stTextArea textarea {
        background: var(--accent-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 0.5rem !important;
    }
    
    .stSelectbox > div > div {
        background: var(--accent-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .stButton button {
        background: var(--accent-color) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid var(--success-color) !important;
        color: var(--success-color) !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid var(--error-color) !important;
        color: var(--error-color) !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid var(--warning-color) !important;
        color: var(--warning-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# Language configurations - simplified
LANGUAGES = {
    'en': {'name': 'English', 'flag': '🇺🇸', 'voices': ['us', 'co.uk', 'com.au']},
    'es': {'name': 'Spanish', 'flag': '🇪🇸', 'voices': ['es']},
    'fr': {'name': 'French', 'flag': '🇫🇷', 'voices': ['fr']},
    'de': {'name': 'German', 'flag': '🇩🇪', 'voices': ['de']},
    'it': {'name': 'Italian', 'flag': '🇮🇹', 'voices': ['it']},
    'pt': {'name': 'Portuguese', 'flag': '🇧🇷', 'voices': ['com.br']},
    'ru': {'name': 'Russian', 'flag': '🇷🇺', 'voices': ['ru']},
    'ja': {'name': 'Japanese', 'flag': '🇯🇵', 'voices': ['co.jp']},
    'zh': {'name': 'Chinese', 'flag': '🇨🇳', 'voices': ['cn']},
    'hi': {'name': 'Hindi', 'flag': '🇮🇳', 'voices': ['co.in']},
}

# Tone templates - no AI required
TONE_TRANSFORMATIONS = {
    'Neutral': {
        'description': 'Clear and professional narration',
        'transform': lambda text: text
    },
    'Dramatic': {
        'description': 'Adds suspense and emphasis',
        'transform': lambda text: add_dramatic_elements(text)
    },
    'Inspiring': {
        'description': 'Uplifting and motivational',
        'transform': lambda text: add_inspiring_elements(text)
    }
}

# Initialize session state
session_vars = ['original_text', 'processed_text', 'detected_lang', 'history', 'selected_tone']
for var in session_vars:
    if var not in st.session_state:
        if var == 'history':
            st.session_state[var] = []
        elif var == 'selected_tone':
            st.session_state[var] = 'Neutral'
        else:
            st.session_state[var] = ''

# Simple language detection using patterns
def detect_language_simple(text):
    """Basic language detection without external dependencies"""
    if not text.strip():
        return 'en', 0.5
    
    text = text.lower()
    
    # Simple pattern matching
    patterns = {
        'zh': r'[\u4e00-\u9fff]',  # Chinese characters
        'ja': r'[\u3040-\u309f\u30a0-\u30ff]',  # Hiragana/Katakana
        'ru': r'[\u0400-\u04ff]',  # Cyrillic
        'hi': r'[\u0900-\u097f]',  # Devanagari
        'ar': r'[\u0600-\u06ff]',  # Arabic
    }
    
    for lang, pattern in patterns.items():
        if re.search(pattern, text):
            return lang, 0.9
    
    # Simple word-based detection for European languages
    word_indicators = {
        'es': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'una', 'del', 'los', 'se', 'las'],
        'fr': ['le', 'de', 'et', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus', 'par'],
        'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als', 'auch', 'es', 'an'],
        'it': ['il', 'di', 'che', 'e', 'la', 'un', 'a', 'per', 'non', 'in', 'una', 'si', 'mi', 'con', 'lo', 'ma', 'me', 'tutto', 'te', 'le', 'da', 'fare', 'era', 'lei'],
        'pt': ['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas']
    }
    
    words = text.split()
    for lang, indicators in word_indicators.items():
        matches = sum(1 for word in words if word in indicators)
        if matches > len(words) * 0.1:  # 10% threshold
            return lang, min(0.9, matches / len(words) * 2)
    
    return 'en', 0.7  # Default to English

# Simple sentence splitting without NLTK
def split_sentences(text):
    """Split text into sentences without NLTK"""
    # Simple sentence ending patterns
    sentences = re.split(r'[.!?]+\s+', text.strip())
    return [s.strip() + '.' for s in sentences if s.strip()]

# Tone transformation functions
def add_dramatic_elements(text):
    """Add dramatic elements to text"""
    sentences = split_sentences(text)
    transformed = []
    
    for sentence in sentences:
        # Add emphasis and pauses
        sentence = sentence.replace('.', '...')
        # Emphasize key words
        words = sentence.split()
        for i, word in enumerate(words):
            if word.lower() in ['suddenly', 'never', 'always', 'everyone', 'nothing', 'everything']:
                words[i] = word.upper()
        transformed.append(' '.join(words))
    
    return ' '.join(transformed)

def add_inspiring_elements(text):
    """Add inspiring elements to text"""
    sentences = split_sentences(text)
    transformed = []
    
    inspiring_replacements = {
        'can': 'CAN',
        'will': 'WILL',
        'possible': 'POSSIBLE',
        'achieve': 'ACHIEVE',
        'success': 'SUCCESS',
        'dream': 'DREAM'
    }
    
    for sentence in sentences:
        # Replace inspiring words with emphasized versions
        words = sentence.split()
        for i, word in enumerate(words):
            clean_word = word.lower().strip('.,!?')
            if clean_word in inspiring_replacements:
                words[i] = word.replace(clean_word, inspiring_replacements[clean_word])
        
        # Change periods to exclamation marks for more energy
        sentence = ' '.join(words).replace('.', '!')
        transformed.append(sentence)
    
    return ' '.join(transformed)

# Text-to-speech function
def generate_audio(text, language='en', voice_region='us', slow_speed=False):
    """Generate audio using gTTS"""
    if not GTTS_AVAILABLE:
        return None
        
    if not text.strip():
        return None
    
    try:
        # Limit text length for performance
        if len(text) > 500:
            text = text[:500] + "..."
            st.warning("Text truncated to 500 characters for faster processing")
        
        tts = gTTS(text=text, lang=language, tld=voice_region, slow=slow_speed)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
        
    except Exception as e:
        st.error(f"Audio generation failed: {str(e)}")
        return None

# Main UI
st.markdown("""
<div class="main-header">
    🎤 EchoVerse Studio - Minimal Edition
</div>
""", unsafe_allow_html=True)

# System status
st.markdown("### System Status")
col1, col2 = st.columns(2)

with col1:
    if GTTS_AVAILABLE:
        st.markdown('<div class="status-card">✅ Text-to-Speech: Available</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-card error">❌ Text-to-Speech: Install gTTS</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="status-card">✅ Language Detection: Built-in</div>', unsafe_allow_html=True)

# Main interface
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">📝 Text Input</h3>', unsafe_allow_html=True)
    
    # Text input
    input_text = st.text_area(
        "Enter your text:",
        height=150,
        placeholder="Enter text in any language... (max 500 characters for optimal performance)",
        max_chars=500,
        value=st.session_state.original_text
    )
    
    # Update session state and detect language
    if input_text != st.session_state.original_text:
        st.session_state.original_text = input_text
        if input_text.strip():
            detected_lang, confidence = detect_language_simple(input_text)
            st.session_state.detected_lang = detected_lang
            
            if detected_lang in LANGUAGES:
                lang_info = LANGUAGES[detected_lang]
                st.markdown(f'<span class="language-badge">{lang_info["flag"]} {lang_info["name"]} ({confidence:.0%})</span>', 
                          unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Settings
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">⚙️ Settings</h3>', unsafe_allow_html=True)
    
    # Target language
    lang_options = [(code, f"{info['flag']} {info['name']}") for code, info in LANGUAGES.items()]
    target_lang = st.selectbox(
        "Target Language:",
        options=lang_options,
        format_func=lambda x: x[1]
    )[0]
    
    # Voice settings
    col_voice1, col_voice2 = st.columns(2)
    with col_voice1:
        voice_region = st.selectbox(
            "Voice Region:",
            options=LANGUAGES[target_lang]['voices']
        )
    
    with col_voice2:
        speech_speed = st.selectbox(
            "Speech Speed:",
            options=[("normal", "Normal"), ("slow", "Slow & Clear")],
            format_func=lambda x: x[1]
        )[0]
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tone selection
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">🎭 Tone Selection</h3>', unsafe_allow_html=True)
    
    tone_cols = st.columns(3)
    for i, (tone, info) in enumerate(TONE_TRANSFORMATIONS.items()):
        with tone_cols[i]:
            if st.button(f"{tone}", key=f"tone_{tone}", use_container_width=True):
                st.session_state.selected_tone = tone
    
    st.info(f"Selected: **{st.session_state.selected_tone}** - {TONE_TRANSFORMATIONS[st.session_state.selected_tone]['description']}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate button
    if st.button("🎨 Generate Audio", type="primary", use_container_width=True):
        if not st.session_state.original_text.strip():
            st.error("Please enter some text first.")
        elif not GTTS_AVAILABLE:
            st.error("Please install gTTS: pip install gtts")
        else:
            with st.spinner("Generating audio..."):
                # Apply tone transformation
                tone_transform = TONE_TRANSFORMATIONS[st.session_state.selected_tone]['transform']
                processed_text = tone_transform(st.session_state.original_text)
                st.session_state.processed_text = processed_text
                
                # Generate audio
                is_slow = speech_speed == "slow"
                audio_data = generate_audio(processed_text, target_lang, voice_region, is_slow)
                
                if audio_data:
                    st.session_state.audio_data = audio_data.read()
                    
                    # Add to history
                    history_entry = {
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'text_preview': st.session_state.original_text[:30] + "...",
                        'language': LANGUAGES[target_lang]['name'],
                        'tone': st.session_state.selected_tone
                    }
                    st.session_state.history.insert(0, history_entry)
                    if len(st.session_state.history) > 5:
                        st.session_state.history = st.session_state.history[:5]
                    
                    st.success("Audio generated successfully!")
                else:
                    st.error("Failed to generate audio. Please try again.")

with col_right:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">👁️ Text Preview</h3>', unsafe_allow_html=True)
    
    # Text preview tabs
    if st.session_state.original_text:
        tab1, tab2 = st.tabs(["Original", "Processed"])
        
        with tab1:
            st.markdown(f'<div class="text-preview-box">{st.session_state.original_text}</div>', 
                       unsafe_allow_html=True)
        
        with tab2:
            if st.session_state.processed_text:
                st.markdown(f'<div class="text-preview-box">{st.session_state.processed_text}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="text-preview-box"><div class="empty-state">Processed text will appear here</div></div>', 
                           unsafe_allow_html=True)
    else:
        st.markdown('<div class="text-preview-box"><div class="empty-state">Enter text to see preview</div></div>', 
                   unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Audio output
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">🔊 Audio Output</h3>', unsafe_allow_html=True)
    
    if 'audio_data' in st.session_state and st.session_state.audio_data:
        st.audio(st.session_state.audio_data, format="audio/mp3")
        
        # Download button
        filename = f"echoverse_{target_lang}_{st.session_state.selected_tone.lower()}.mp3"
        st.download_button(
            "💾 Download Audio",
            data=st.session_state.audio_data,
            file_name=filename,
            mime="audio/mp3"
        )
    else:
        st.info("Generated audio will appear here")
    
    st.markdown('</div>', unsafe_allow_html=True)

# History section
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📚 Recent Generations")
    
    for entry in st.session_state.history:
        st.write(f"**{entry['timestamp']}** - {entry['language']} - {entry['tone']}")
        st.caption(entry['text_preview'])

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-secondary); padding: 1rem;">
    <strong>EchoVerse Studio - Minimal Edition</strong><br>
    Lightweight text-to-speech without heavy dependencies<br>
    <em>No TensorFlow • No NLTK • No Transformers</em>
</div>
""", unsafe_allow_html=True)