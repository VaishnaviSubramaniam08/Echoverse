from transformers import pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rewrite_text(input_text, tone="Neutral"):
    try:
        logger.info("Loading distilgpt2 model...")
        generator = pipeline("text-generation", model="distilgpt2")
        logger.info("Model loaded successfully")
        prompt = f"Rewrite this sentence in a {tone} tone while preserving its meaning: {input_text}"
        logger.info(f"Generating text with prompt: {prompt}")
        result = generator(prompt, max_length=50, num_return_sequences=1)
        logger.info("Text generated")
        return result[0]['generated_text']
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    text = "The train arrived at the station at sunrise."
    rewritten = rewrite_text(text, tone="Suspenseful")
    print("Original:", text)
    print("Rewritten:", rewritten)