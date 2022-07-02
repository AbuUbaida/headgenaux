import os
import datasets
import transformers
import pandas as pd
import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from transformers import (BertTokenizerFast, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EncoderDecoderModel,
                            get_linear_schedule_with_warmup)
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

dataset = load_dataset('amazon_reviews_multi', 'en', split='train')
dataset = dataset.train_test_split(test_size=0.1)
val_data = dataset['test'].train_test_split(test_size=0.5)
dataset = DatasetDict({
    'train': dataset['train'],
    'valid': val_data['train'],
    'test': val_data['test'],})
train_data = dataset['train']

batch_size = 4
encoder_max_length = 512
decoder_max_length = 128

def process_data_to_model_inputs(batch):
    inputs = tokenizer(batch['review_body'], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch['review_title'], padding="max_length", truncation=True, max_length=decoder_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

train_data = train_data.select(range(32))

train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=['language', 'product_category','review_body','review_title', 'product_id', 'review_id', 'reviewer_id', 'stars']
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=4)

model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size
model.config.max_length = 142
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.config.early_stopping = True
model.config.length_penalty = 2.0
model.config.num_beams = 4

num_training_steps = len(train_dataloader) * 40

device = xm.xla_device()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
print('Device: ', device)



for epoch in range(30):
    print(f"Epoch:", epoch)

    for batch in train_dataloader:
        model.train()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        print("Loss:", loss.item())
        loss.backward()
        # optimizer.step()
        xm.optimizer_step(optimizer, barrier=True)
        # optimizer.step()
        # xm.mark_step()
        scheduler.step()