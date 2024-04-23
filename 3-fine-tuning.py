import evaluate
import numpy as np
import torch

from datasets     import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer


def tokenize_fn(input):
    return tokenizer(input["sentence1"], input["sentence2"], truncation=True)


def compute_metrics(eval_preds):
    metric         = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    preds          = np.argmax(logits, axis=-1)

    return metric.compute(predictions=preds, references=labels)


raw_datasets = load_dataset("glue", "mrpc")

checkpoint         = "bert-base-uncased"
tokenizer          = AutoTokenizer.from_pretrained(checkpoint)
model              = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)
data_collator      = DataCollatorWithPadding(tokenizer=tokenizer)
training_args      = TrainingArguments("test-trainer", evaluation_strategy="epoch")
data_collator      = DataCollatorWithPadding(tokenizer=tokenizer)
trainer            = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Training...")
trainer.train()
print("Training done!")
