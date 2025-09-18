from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_name = "ibm-granite/granite-3.3-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
print("Model loaded successfully!")