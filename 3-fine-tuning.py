import torch

from datasets     import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer


def tokenize_fn(input):
    return tokenizer(input["sentence1"], input["sentence2"], truncation=True)


raw_datasets = load_dataset("glue", "mrpc")

checkpoint         = "bert-base-uncased"
tokenizer          = AutoTokenizer.from_pretrained(checkpoint)
model              = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)
data_collator      = DataCollatorWithPadding(tokenizer=tokenizer)
training_args      = TrainingArguments("test-trainer")
data_collator      = DataCollatorWithPadding(tokenizer=tokenizer)
trainer            = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

print("Training...")
trainer.train()
print("Training done!")
