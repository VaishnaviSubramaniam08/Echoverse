import streamlit as st
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS, gTTSError
import pyttsx3
from io import BytesIO
import base64

# Initialize the model and tokenizer for text generation
@st.cache_resource
def load_model():
    try:
        model_name = "ibm-granite/granite-3.3-2b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # CPU
        return pipe, tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None, None

pipe, tokenizer, model = load_model()
if pipe is None:
    st.stop()

# Initialize pyttsx3 engine for fallback
@st.cache_resource
def init_pyttsx3():
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        voice_map = {
            "Lisa (en-us)": None,
            "Michael (en-uk)": None,
            "Allison (en-au)": None
        }
        for v in voices:
            if "US" in v.name and not voice_map["Lisa (en-us)"]:
                voice_map["Lisa (en-us)"] = v.id
            elif "UK" in v.name and not voice_map["Michael (en-uk)"]:
                voice_map["Michael (en-uk)"] = v.id
            elif "AU" in v.name and not voice_map["Allison (en-au)"]:
                voice_map["Allison (en-au)"] = v.id
        return engine, voice_map
    except Exception as e:
        st.error(f"Failed to initialize pyttsx3: {str(e)}")
        return None, None

pyttsx3_engine, pyttsx3_voice_map = init_pyttsx3()
if pyttsx3_engine is None:
    st.stop()

# Function to rewrite text with a specific tone
def rewrite_text(text, tone):
    prompt = f"""
    You are an expert writer tasked with rewriting the following text in a {tone} tone while preserving its original meaning and enhancing its expressiveness. Ensure the rewritten text is concise, natural, and suitable for narration. Here is the text:

    {text}

    Provide only the rewritten text as the output.
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        response = pipe(messages, max_new_tokens=500, num_return_sequences=1)
        rewritten_text = response[0]["generated_text"][-1]["content"]
        return rewritten_text.strip()
    except Exception as e:
        st.error(f"Text generation failed: {str(e)}")
        return None

# Function to convert text to speech with fallback
def text_to_speech(text, voice_tld, voice_name):
    audio_fp = BytesIO()
    tlds = [voice_tld, "com", "co.uk", "com.au"]
    gtts_success = False

    for tld in tlds:
        try:
            tts = gTTS(text=text, lang="en", tld=tld)
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            gtts_success = True
            st.info(f"Audio generated using gTTS with {tld} voice.")
            break
        except gTTSError as e:
            st.warning(f"gTTS failed with TLD {tld}: {str(e)}")
            continue

    if not gtts_success:
        try:
            pyttsx3_engine.setProperty('voice', pyttsx3_voice_map.get(voice_name, pyttsx3_voice_map["Lisa (en-us)"]))
            pyttsx3_engine.save_to_file(text, "temp_audio.wav")
            pyttsx3_engine.runAndWait()
            with open("temp_audio.wav", "rb") as f:
                audio_fp.write(f.read())
            audio_fp.seek(0)
            st.info("Audio generated using pyttsx3 (offline fallback).")
            os.remove("temp_audio.wav")
        except Exception as e:
            st.error(f"Failed to generate audio with pyttsx3: {str(e)}")
            return None

    return audio_fp

# Function to convert audio to base64 for streaming
def audio_to_base64(audio_fp):
    if audio_fp is None:
        return None
    audio_fp.seek(0)
    audio_bytes = audio_fp.read()
    b64 = base64.b64encode(audio_bytes).decode()
    mime = "audio/mp3" if audio_fp.getvalue().startswith(b"\xFF\xFB") else "audio/wav"
    return f"data:{mime};base64,{b64}"

# Streamlit UI
st.set_page_config(page_title="EchoVerse", page_icon="🎙️", layout="wide")
st.title("EchoVerse: AI Audiobook Creator")
st.write("Transform your text into expressive audio with customizable tones.")

# Input section
st.subheader("Input Text")
input_method = st.radio("Choose input method:", ("Paste Text", "Upload .txt File"))
input_text = ""
if input_method == "Paste Text":
    input_text = st.text_area("Paste your text here:", height=200, placeholder="Enter your text here...")
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded_file:
        input_text = uploaded_file.read().decode("utf-8")
        st.text_area("Uploaded Text:", value=input_text, height=200, disabled=True)

# Tone and voice selection
st.subheader("Customize Output")
col1, col2 = st.columns(2)
with col1:
    tone = st.selectbox("Select Tone:", ["Neutral", "Suspenseful", "Inspiring"])
with col2:
    voice = st.selectbox("Select Voice:", ["Lisa (en-us)", "Michael (en-uk)", "Allison (en-au)"])
voice_map = {"Lisa (en-us)": "us", "Michael (en-uk)": "co.uk", "Allison (en-au)": "com.au"}

# Process button
if st.button("Generate Audiobook", key="generate"):
    if not input_text.strip():
        st.error("Please provide text input.")
    else:
        with st.spinner("Rewriting text in {} tone...".format(tone)):
            rewritten_text = rewrite_text(input_text, tone)
            if rewritten_text is None:
                st.stop()

        st.subheader("Text Comparison")
        col3, col4 = st.columns(2)
        with col3:
            st.write("**Original Text**")
            st.text_area("Original", value=input_text, height=200, disabled=True)
        with col4:
            st.write(f"**{tone} Text**")
            st.text_area(f"{tone} Text", value=rewritten_text, height=200, disabled=True)

        with st.spinner("Generating audio..."):
            audio_fp = text_to_speech(rewritten_text, voice_map[voice], voice)
            audio_base64 = audio_to_base64(audio_fp)

            if audio_base64:
                st.subheader("Audio Output")
                st.audio(audio_base64, format=audio_base64.split(";")[0].split(":")[1])
                audio_fp.seek(0)
                mime = "audio/mp3" if audio_fp.getvalue().startswith(b"\xFF\xFB") else "audio/wav"
                st.download_button(
                    label="Download Audio",
                    data=audio_fp,
                    file_name=f"echoverse_{tone.lower()}_audio.{mime.split('/')[-1]}",
                    mime=mime,
                    key="download"
                )
            else:
                st.error("Audio generation failed. Please check your internet connection or try again.")

# Instructions
st.markdown("""
### How to Use EchoVerse
1. **Input Text**: Paste text or upload a .txt file.
2. **Select Tone**: Choose Neutral, Suspenseful, or Inspiring.
3. **Select Voice**: Choose a voice (Lisa, Michael, or Allison).
4. **Generate**: Click the "Generate Audiobook" button to rewrite text and create audio.
5. **Compare & Listen**: View the original and rewritten text side-by-side, then listen to or download the audio.

**Note**: If audio generation fails, the app will attempt an offline fallback using pyttsx3.
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Hugging Face Transformers, gTTS, and pyttsx3")