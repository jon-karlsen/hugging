import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def pad_arr(arr, start, len, pad):
    count = 0

    for _ in range(start, len):
        arr.append(pad)
        count += 1

    return count


def gen_mask(len):
    res = []

    for _ in range(len):
        res.append(1)

    return res


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence_1 = "I've been watiting for a HuggingFace course my entire life."
sequence_2 = "I hate this so much!"

sequence1_tokens = tokenizer.tokenize(sequence_1)
sequence2_tokens = tokenizer.tokenize(sequence_2)

sequence1_ids = tokenizer.convert_tokens_to_ids(sequence1_tokens)
sequence2_ids = tokenizer.convert_tokens_to_ids(sequence2_tokens)

mask_1 = gen_mask(len(sequence1_ids))
mask_2 = gen_mask(len(sequence2_ids))

pad_arr(sequence2_ids, len(sequence2_ids) - 1, len(sequence1_ids) - 1, tokenizer.pad_token_id)
pad_arr(mask_2, len(mask_2) - 1, len(mask_1) - 1, 0)

batched_ids = [
    sequence1_ids,
    sequence2_ids
]

attention_mask = [mask_1, mask_2]

print(model(torch.tensor([tokenizer.convert_tokens_to_ids(sequence1_tokens)])).logits)
print(model(torch.tensor([tokenizer.convert_tokens_to_ids(sequence2_tokens)])).logits)
print(
    model(
        torch.tensor(batched_ids),
        attention_mask=torch.tensor(attention_mask)
    ).logits
)
