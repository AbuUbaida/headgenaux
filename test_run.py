import datasets
import transformers
import evaluate
import pandas as pd
import numpy as np
import torch
import torch_xla.core.xla_model as xm
from tqdm.notebook import tqdm
from transformers import BertTokenizerFast, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EncoderDecoderModel
from normalizer import normalize
from datasets import load_dataset, DatasetDict, Dataset, ClassLabel
from bangla_rouge_score import rouge_scorer, scoring
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
train_data = load_dataset('csv', data_files='data/train_data_captioned.csv', split='train')
val_data = load_dataset('csv', data_files='data/valid_data_captioned.csv', split='train')

batch_size = 4
encoder_max_length = 512
decoder_max_length = 128

def process_data_to_model_inputs(batch):
    for idx, example in enumerate(batch['headline']):
        batch['headline'][idx] = normalize(example)
    for idx, example in enumerate(batch['article']):
        batch['article'][idx] = normalize(example)
        
    # tokenize the inputs and labels 
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["headline"], padding="max_length", truncation=True, max_length=decoder_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

train_data = train_data.select(range(32))

train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["article", "headline", "ic1"]
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)

val_data = val_data.select(range(8))

val_data = val_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["article", "headline", "ic1"]
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=4)
val_dataloader = DataLoader(val_data, batch_size=4)

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

def rouge_compute(predictions, references, rouge_types=None, use_aggregator=True, use_stemmer=True):
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer, lang="bengali")
    if use_aggregator:
        aggregator = scoring.BootstrapAggregator()
    else:
        scores = []

    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        if use_aggregator:
            aggregator.add_scores(score)
        else:
            scores.append(score)

    if use_aggregator:
        result = aggregator.aggregate()
    else:
        result = {}
        for key in scores[0]:
            result[key] = list(score[key] for score in scores)

    return result

def rouge_score(pred_str, label_str):
    rouge_score = rouge_compute(predictions=pred_str, references=label_str, rouge_types=["rouge1", "rouge2", "rougeL"])
    rouge1 = rouge_score['rouge1'].mid
    rouge2 = rouge_score['rouge2'].mid
    rougeL = rouge_score['rougeL'].mid
    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL
    }

def compute_metrics(pred_ids, label_ids):
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids==-100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    rouge_output = rouge_score(pred_str, label_str)

    return {
        "rouge1_precision": round(rouge_output["rouge1"].precision, 4),
        "rouge1_recall": round(rouge_output["rouge1"].recall, 4),
        "rouge1_fmeasure": round(rouge_output["rouge1"].fmeasure, 4),
        "rouge2_precision": round(rouge_output["rouge2"].precision, 4),
        "rouge2_recall": round(rouge_output["rouge2"].recall, 4),
        "rouge2_fmeasure": round(rouge_output["rouge2"].fmeasure, 4),
        "rougeL_precision": round(rouge_output["rougeL"].precision, 4),
        "rougeL_recall": round(rouge_output["rougeL"].recall, 4),
        "rougeL_fmeasure": round(rouge_output["rougeL"].fmeasure, 4)
    }

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = xm.xla_device()
model.to(device)
print('Device: ', device)

for epoch in range(40):  # loop over the dataset multiple times
    print(f"Epoch:", epoch)
    # train + evaluate on training data
    running_loss = 0.0
    train_rouge_1_precision = 0.0
    train_rouge_1_recall = 0.0
    train_rouge_1_f1 = 0.0
    train_rouge_2_precision = 0.0
    train_rouge_2_recall = 0.0
    train_rouge_2_f1 = 0.0
    train_rouge_L_precision = 0.0
    train_rouge_L_recall = 0.0
    train_rouge_L_f1 = 0.0

    for batch in tqdm(train_dataloader):
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
        running_loss += loss.item()
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)

        print("Running Loss:", loss.item())

        # evaluate (batch generation)
        # model.eval()
        # outputs = model.generate(batch["input_ids"].to(device))
        # compute metrics
        # metrics = compute_metrics(pred_ids=outputs, label_ids=batch["labels"])
        # train_rouge_1_precision += metrics["rouge1_precision"] 
        # train_rouge_1_recall += metrics["rouge1_recall"] 
        # train_rouge_1_f1 += metrics["rouge1_fmeasure"]
        # train_rouge_2_precision += metrics["rouge2_precision"] 
        # train_rouge_2_recall += metrics["rouge2_recall"]  
        # train_rouge_2_f1 += metrics["rouge2_fmeasure"] 
        # train_rouge_L_precision += metrics["rougeL_precision"] 
        # train_rouge_L_recall += metrics["rougeL_recall"] 
        # train_rouge_L_f1 += metrics["rougeL_fmeasure"]
  
    # print("Loss: ", running_loss / len(train_dataloader))
    # print("Train ROUGE 1 precision:", train_rouge_1_precision / len(train_dataloader))
    # print("Train ROUGE 1 recall:", train_rouge_1_recall / len(train_dataloader))
    # print("Train ROUGE 1 F1:", train_rouge_1_f1 / len(train_dataloader))
    # print("Train ROUGE 2 precision:", train_rouge_2_precision / len(train_dataloader))
    # print("Train ROUGE 2 recall:", train_rouge_2_recall / len(train_dataloader))
    # print("Train ROUGE 2 F1:", train_rouge_2_f1 / len(train_dataloader))
    # print("Train ROUGE L precision:", train_rouge_L_precision / len(train_dataloader))
    # print("Train ROUGE L recall:", train_rouge_L_recall / len(train_dataloader))
    # print("Train ROUGE L F1:", train_rouge_L_f1 / len(train_dataloader))