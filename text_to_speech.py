import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import requests
import json
import base64
import io
import os
from datetime import datetime
import tempfile
import zipfile

# Initialize the app
st.set_page_config(
    page_title="EchoVerse - AI Audiobook Creator",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .text-comparison {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
    
    .audio-player {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = ""
if 'original_text' not in st.session_state:
    st.session_state.original_text = ""
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Load model function with caching
@st.cache_resource
def load_granite_model():
    """Load the IBM Granite model with caching"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "ibm-granite/granite-3.3-2b-instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def rewrite_text_with_tone(text, tone, tokenizer, model):
    """Rewrite text with specified tone using IBM Granite model"""
    
    tone_prompts = {
        "Neutral": "Rewrite the following text in a clear, neutral, and professional tone while preserving all the original meaning and information:",
        "Suspenseful": "Rewrite the following text in a suspenseful and engaging tone that builds tension and keeps readers on edge, while preserving all the original meaning:",
        "Inspiring": "Rewrite the following text in an inspiring and motivational tone that uplifts and energizes readers, while preserving all the original meaning:"
    }
    
    prompt = tone_prompts.get(tone, tone_prompts["Neutral"])
    
    messages = [
        {"role": "user", "content": f"{prompt}\n\nOriginal text: {text}"}
    ]
    
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=min(len(text.split()) * 2, 512),
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        rewritten_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return rewritten_text.strip()
        
    except Exception as e:
        st.error(f"Error in text rewriting: {str(e)}")
        return text

def text_to_speech_mock(text, voice="Lisa"):
    """Mock TTS function - In real implementation, integrate with IBM Watson TTS"""
    # This is a placeholder for IBM Watson Text-to-Speech integration
    # You would replace this with actual IBM Watson TTS API calls
    
    st.info(f"🎵 Generating audio with {voice} voice...")
    st.info("Note: This is a mock implementation. In production, integrate with IBM Watson Text-to-Speech API.")
    
    # Mock audio data (in real implementation, this would be actual audio from IBM Watson)
    mock_audio_info = {
        "voice": voice,
        "text_length": len(text),
        "estimated_duration": f"{len(text.split()) * 0.5:.1f} seconds",
        "format": "MP3"
    }
    
    return mock_audio_info

def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 EchoVerse</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Audiobook Creation System</p>', unsafe_allow_html=True)
    
    # Features overview
    with st.expander("✨ Features Overview", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 Core Features:**
            - Tone-Adaptive Text Rewriting
            - High-Quality Voice Narration
            - Downloadable Audio Output
            - Side-by-Side Text Comparison
            """)
        
        with col2:
            st.markdown("""
            **🔧 Technology Stack:**
            - IBM Granite LLM for text rewriting
            - IBM Watson Text-to-Speech
            - Streamlit for user interface
            - Python backend processing
            """)
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading IBM Granite model... This may take a few minutes."):
            tokenizer, model = load_granite_model()
            if tokenizer and model:
                st.session_state.tokenizer = tokenizer
                st.session_state.model = model
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model. Please check your setup.")
                return
    
    # Input section
    st.markdown("## 📝 Input Your Text")
    
    input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"], horizontal=True)
    
    if input_method == "Paste Text":
        user_text = st.text_area(
            "Enter your text here:",
            height=200,
            placeholder="Paste your text content here..."
        )
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
        user_text = ""
        if uploaded_file:
            user_text = str(uploaded_file.read(), "utf-8")
            st.text_area("Uploaded content:", value=user_text, height=200, disabled=True)
    
    if user_text:
        st.session_state.original_text = user_text
        
        # Tone selection
        st.markdown("## 🎭 Select Tone")
        tone = st.selectbox(
            "Choose the tone for rewriting:",
            ["Neutral", "Suspenseful", "Inspiring"],
            help="Select how you want the text to be rewritten"
        )
        
        # Voice selection
        st.markdown("## 🎤 Select Voice")
        col1, col2 = st.columns(2)
        with col1:
            voice = st.selectbox(
                "Choose narrator voice:",
                ["Lisa", "Michael", "Allison"],
                help="Select the voice for text-to-speech conversion"
            )
        
        # Process button
        if st.button("🚀 Generate Audiobook", type="primary"):
            with st.spinner("Processing your text..."):
                # Step 1: Rewrite text with selected tone
                st.info("Step 1/2: Rewriting text with selected tone...")
                rewritten_text = rewrite_text_with_tone(
                    user_text, 
                    tone, 
                    st.session_state.tokenizer, 
                    st.session_state.model
                )
                st.session_state.processed_text = rewritten_text
                
                # Step 2: Convert to speech
                st.info("Step 2/2: Converting to speech...")
                audio_info = text_to_speech_mock(rewritten_text, voice)
                st.session_state.audio_data = audio_info
                
                st.success("✅ Audiobook generated successfully!")
        
        # Display results
        if st.session_state.processed_text:
            st.markdown("## 📊 Text Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Text")
                st.markdown(f'<div class="text-comparison">{st.session_state.original_text}</div>', 
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"### {tone} Tone")
                st.markdown(f'<div class="text-comparison">{st.session_state.processed_text}</div>', 
                           unsafe_allow_html=True)
            
            # Audio section
            if st.session_state.audio_data:
                st.markdown("## 🎧 Generated Audio")
                
                audio_info = st.session_state.audio_data
                
                st.markdown(f"""
                <div class="audio-player">
                    <h3>🎵 Audio Ready!</h3>
                    <p><strong>Voice:</strong> {audio_info['voice']}</p>
                    <p><strong>Estimated Duration:</strong> {audio_info['estimated_duration']}</p>
                    <p><strong>Format:</strong> {audio_info['format']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.button("🎵 Play Audio", help="Mock button - would play audio in real implementation")
                
                with col2:
                    st.button("⏸️ Pause", help="Mock button - would pause audio in real implementation")
                
                with col3:
                    st.button("📥 Download MP3", help="Mock button - would download audio in real implementation")
                
                # Statistics
                st.markdown("### 📈 Generation Statistics")
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                with stats_col1:
                    st.metric("Original Words", len(st.session_state.original_text.split()))
                
                with stats_col2:
                    st.metric("Processed Words", len(st.session_state.processed_text.split()))
                
                with stats_col3:
                    st.metric("Processing Time", "~30 seconds")
    
    # Sidebar with instructions
    with st.sidebar:
        st.markdown("## 📚 How to Use")
        st.markdown("""
        1. **Input Text**: Paste or upload your text
        2. **Select Tone**: Choose from Neutral, Suspenseful, or Inspiring
        3. **Choose Voice**: Pick your preferred narrator
        4. **Generate**: Click to create your audiobook
        5. **Compare**: Review original vs. rewritten text
        6. **Listen**: Play and download your audio
        """)
        
        st.markdown("## ⚙️ Technical Setup")
        st.markdown("""
        **Required Dependencies:**
        ```bash
        pip install streamlit transformers torch
        pip install requests ibm-watson
        ```
        
        **IBM Watson Setup:**
        - Get API key from IBM Cloud
        - Configure Text-to-Speech service
        - Add credentials to environment
        """)
        
        st.markdown("## 🎯 Use Cases")
        st.markdown("""
        - **Students**: Convert study materials to audio
        - **Professionals**: Create audio reports
        - **Accessibility**: Help visually impaired users
        - **Content Creation**: Generate engaging narrations
        """)

if __name__ == "__main__":
    main()