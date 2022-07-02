import os
import datasets
import transformers
import pandas as pd
import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import warnings
from transformers import (BertTokenizerFast, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EncoderDecoderModel,
                            get_linear_schedule_with_warmup)
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader
from scipy import stats
warnings.filterwarnings('ignore')

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def process_data_to_model_inputs(batch):
    encoder_max_length = 512
    decoder_max_length = 128
    inputs = tokenizer(batch['review_body'], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch['review_title'], padding="max_length", truncation=True, max_length=decoder_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

def run(index):
    batch_size = 16

    dataset = load_dataset('amazon_reviews_multi', 'en', split='train')
    dataset = dataset.train_test_split(test_size=0.1)
    val_data = dataset['test'].train_test_split(test_size=0.5)
    dataset = DatasetDict({
        'train': dataset['train'],
        'valid': val_data['train'],
        'test': val_data['test'],})
    train_data = dataset['train']

    train_data = train_data.select(range(10000))

    train_data = train_data.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=batch_size, 
        remove_columns=['language', 'product_category','review_body','review_title', 'product_id', 'review_id', 'reviewer_id', 'stars']
    )
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"],
    )
    train_sampler = torch.utils.data.DistributedSampler(
        train_data,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=4)

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


    device = xm.xla_device()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 5e-5*xm.xrt_world_size()
    num_training_steps = int(len(train_dataloader) / xm.xrt_world_size() * 30)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    print('Device: ', device)

    for epoch in range(30):
        print(f"Epoch:", epoch)
        para_loader = pl.ParallelLoader(train_dataloader, [device])
        for batch in para_loader.per_device_loader(device):
            model.train()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            xm.master_print("Loss:", loss.item())
            loss.backward()
            # optimizer.step()
            xm.optimizer_step(optimizer)
            # optimizer.step()
            # xm.mark_step()
            scheduler.step()

if __name__ == "__main__":
    xmp.spawn(run, nprocs=8)