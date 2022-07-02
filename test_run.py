import datasets
import transformers
import evaluate
import pandas as pd
import numpy as np
import torch
import time
import torch_xla.core.xla_model as xm
from tqdm.notebook import tqdm
from transformers import BertTokenizerFast, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EncoderDecoderModel
from normalizer import normalize
from datasets import load_dataset, DatasetDict, Dataset, ClassLabel
from bangla_rouge_score import rouge_scorer, scoring
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
train_data = load_dataset('csv', data_files='data/palokal_merged_with_cap_v1.0.csv', split='train')
# val_data = load_dataset('csv', data_files='data/valid_data_captioned.csv', split='train')

batch_size = 16
encoder_max_length = 512
decoder_max_length = 32

def process_data_to_model_inputs(batch):
    for idx, example in enumerate(batch['head_lines']):
        batch['head_lines'][idx] = normalize(example)
    for idx, example in enumerate(batch['article']):
        batch['article'][idx] = normalize(example)
        
    # tokenize the inputs and labels 
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["head_lines"], padding="max_length", truncation=True, max_length=decoder_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

train_data = train_data.select(range(64))

train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["article", "head_lines", "news_link", "category"]
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)

# val_data = val_data.select(range(8))

# val_data = val_data.map(
#     process_data_to_model_inputs, 
#     batched=True, 
#     batch_size=batch_size, 
#     remove_columns=["article", "headline", "ic1"]
# )
# val_data.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "labels"],
# )

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
# val_dataloader = DataLoader(val_data, batch_size=4)

model = EncoderDecoderModel.from_encoder_decoder_pretrained("csebuetnlp/banglabert", "csebuetnlp/banglabert")
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size
model.config.max_length = 16
model.config.min_length = 4
model.config.no_repeat_ngram_size = 2
model.config.early_stopping = True
model.config.length_penalty = 1
model.config.num_beams = 4

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = xm.xla_device()
model.to(device)
print('Device: ', device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(30):  # loop over the dataset multiple times
    print(f"Epoch:", epoch)
    # train + evaluate on training data
    running_loss = 0.0
    start = time.time()
    for batch in train_dataloader:
        model.train()
        # get the inputs; 
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        # running_loss += loss.item()
        loss.backward()
        # optimizer.step()
        xm.optimizer_step(optimizer, barrier=True)
        # xm.reduce_gradients(optimizer)
        # # Parameter Update
        # optimizer.step()

        print("Running Loss:", loss.item())
    end = time.time()
    print('Elapsed time per epoch: ', end-start)