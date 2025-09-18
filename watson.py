"""
IBM Watson Text-to-Speech Integration for EchoVerse
This file contains the actual implementation for Watson TTS integration
"""

import os
import json
import base64
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import streamlit as st
import io

class WatsonTTSIntegrator:
    def __init__(self, api_key=None, url=None):
        """
        Initialize Watson Text-to-Speech service
        
        Args:
            api_key (str): IBM Watson API key
            url (str): IBM Watson service URL
        """
        self.api_key = api_key or os.getenv('WATSON_TTS_API_KEY')
        self.url = url or os.getenv('WATSON_TTS_URL')
        
        if not self.api_key or not self.url:
            raise ValueError("Watson TTS credentials not found. Set WATSON_TTS_API_KEY and WATSON_TTS_URL environment variables.")
        
        # Initialize authenticator and service
        authenticator = IAMAuthenticator(self.api_key)
        self.text_to_speech = TextToSpeechV1(authenticator=authenticator)
        self.text_to_speech.set_service_url(self.url)
        
        # Voice mapping
        self.voice_mapping = {
            "Lisa": "en-US_LisaV3Voice",
            "Michael": "en-US_MichaelV3Voice", 
            "Allison": "en-US_AllisonV3Voice",
            "Kevin": "en-US_KevinV3Voice",
            "Emily": "en-US_EmilyV3Voice"
        }
    
    def synthesize_text(self, text, voice="Lisa", format="audio/mp3"):
        """
        Convert text to speech using IBM Watson TTS
        
        Args:
            text (str): Text to convert
            voice (str): Voice to use
            format (str): Audio format
            
        Returns:
            dict: Audio data and metadata
        """
        try:
            watson_voice = self.voice_mapping.get(voice, self.voice_mapping["Lisa"])
            
            # Synthesize speech
            response = self.text_to_speech.synthesize(
                text=text,
                voice=watson_voice,
                accept=format
            ).get_result()
            
            # Get audio content
            audio_content = response.content
            
            # Create audio info
            audio_info = {
                "audio_data": audio_content,
                "voice": voice,
                "watson_voice": watson_voice,
                "format": format,
                "text_length": len(text),
                "word_count": len(text.split()),
                "estimated_duration": self._estimate_duration(text),
                "size_bytes": len(audio_content)
            }
            
            return audio_info
            
        except Exception as e:
            st.error(f"Error in Watson TTS: {str(e)}")
            return None
    
    def _estimate_duration(self, text):
        """Estimate audio duration based on text length"""
        words = len(text.split())
        # Average speaking rate: 150-160 words per minute
        duration_minutes = words / 155
        return f"{duration_minutes:.1f} minutes"
    
    def get_available_voices(self):
        """Get list of available voices from Watson"""
        try:
            voices = self.text_to_speech.list_voices().get_result()
            return [voice['name'] for voice in voices['voices']]
        except Exception as e:
            st.error(f"Error fetching voices: {str(e)}")
            return list(self.voice_mapping.keys())

# Configuration setup functions
def setup_watson_credentials():
    """Setup Watson credentials in Streamlit sidebar"""
    st.sidebar.markdown("## 🔐 Watson TTS Configuration")
    
    api_key = st.sidebar.text_input(
        "Watson TTS API Key", 
        type="password",
        help="Enter your IBM Watson Text-to-Speech API key"
    )
    
    url = st.sidebar.text_input(
        "Watson TTS URL",
        value="https://api.us-south.text-to-speech.watson.cloud.ibm.com",
        help="Enter your Watson TTS service URL"
    )
    
    if api_key and url:
        os.environ['WATSON_TTS_API_KEY'] = api_key
        os.environ['WATSON_TTS_URL'] = url
        return True
    
    return False

def create_audio_player(audio_data, filename="echoverse_audio.mp3"):
    """Create audio player widget in Streamlit"""
    if audio_data:
        st.audio(audio_data, format="audio/mp3")
        
        # Download button
        st.download_button(
            label="📥 Download MP3",
            data=audio_data,
            file_name=filename,
            mime="audio/mp3"
        )

# Example usage and testing
def test_watson_integration():
    """Test Watson TTS integration"""
    try:
        # Initialize TTS
        tts = WatsonTTSIntegrator()
        
        # Test text
        test_text = "Hello, this is a test of the IBM Watson Text-to-Speech integration for EchoVerse."
        
        # Synthesize
        result = tts.synthesize_text(test_text, voice="Lisa")
        
        if result:
            st.success("✅ Watson TTS integration working!")
            st.json(result)
            return True
        else:
            st.error("❌ Watson TTS test failed")
            return False
            
    except Exception as e:
        st.error(f"Watson TTS test error: {str(e)}")
        return False

# Enhanced text preprocessing for better TTS output
def preprocess_text_for_tts(text):
    """
    Preprocess text to improve TTS quality
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    import re
    
    # Add pauses for better flow
    text = re.sub(r'\.(?!\s)', '. ', text)  # Ensure space after periods
    text = re.sub(r'\,(?!\s)', ', ', text)  # Ensure space after commas
    text = re.sub(r'\;(?!\s)', '; ', text)  # Ensure space after semicolons
    
    # Handle abbreviations
    abbreviations = {
        'Dr.': 'Doctor',
        'Mr.': 'Mister', 
        'Mrs.': 'Missus',
        'Ms.': 'Miss',
        'Prof.': 'Professor',
        'etc.': 'etcetera',
        'vs.': 'versus',
        'e.g.': 'for example',
        'i.e.': 'that is'
    }
    
    for abbrev, full_form in abbreviations.items():
        text = text.replace(abbrev, full_form)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

if __name__ == "__main__":
    # Example usage
    print("Watson TTS Integration Module")
    print("Use this module in your main EchoVerse application")