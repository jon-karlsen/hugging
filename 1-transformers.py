from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Architecture: skeleton of model; definition of each layer and each operation that happens within model
# Checkpoint: weights that will be loadded in a given architecture
# Model: umbrella term less precise than "architecture" or "checkpoint"; depending on context, can mean both.

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

print(model.config.id2label)

raw_inputs = [
    "I finally have a reason to start working with AI.",
    "I wish the Korean summer were at least not quite so humid."
]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

print(predictions)
# print(inputs)
