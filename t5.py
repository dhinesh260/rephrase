from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load tokenizer and model (must match)
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Example text
text = "paraphrase: The server is down. Restart it."

# Tokenize the input
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

# Generate paraphrased text
outputs = model.generate(**inputs, max_length=512)
paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Paraphrased:", paraphrased_text)
