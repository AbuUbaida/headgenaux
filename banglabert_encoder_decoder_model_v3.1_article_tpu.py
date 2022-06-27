from multiprocessing import Barrier
import datasets
import transformers
import evaluate
import pandas as pd
import numpy as np
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer, EncoderDecoderModel
from normalizer import normalize
from datasets import load_dataset, DatasetDict
from nltk.translate import meteor_score
from bangla_rouge_score import rouge_scorer, scoring
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

destination_folder = "base-model"

train_data = load_dataset('csv', data_files='Data/train_data_captioned.csv', split='train')
val_data = load_dataset('csv', data_files='Data/valid_data_captioned.csv', split='train')

tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")

batch_size = 16
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

train_data = train_data.select(range(64))
val_data = val_data.select(range(8))

train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["article", "headline", "ic1"]
)
val_data = val_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["article", "headline", "ic1"]
)

train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
val_dataloader = DataLoader(val_data, batch_size=16)

model = EncoderDecoderModel.from_encoder_decoder_pretrained("csebuetnlp/banglabert", "csebuetnlp/banglabert")

# set special tokens
'''
decoder_start_token_id (int, optional) — If an encoder-decoder model starts decoding with a different token than bos,
                the id of that token.

pad_token_id (int, optional) — The id of the padding token.

eos_token_id (int, optional) — The id of the end-of-stream token.

vocab_size (int) — The number of tokens in the vocabulary, which is also the first dimension of the embeddingsmatrix
                (this attribute may be missing for models that don’t have a text modality like ViT).
'''
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# sensible parameters for beam search
'''
max_length (int, optional, defaults to 20) — Maximum length that will be used by default in the
                generate method of the model.

min_length (int, optional, defaults to 10) — Minimum length that will be used by default in the
                generate method of the model.

no_repeat_ngram_size (int, optional, defaults to 0) — Value that will be used by default in the — generate method
                of the model for no_repeat_ngram_size. If set to int > 0, all ngrams of that size can only occur once.

early_stopping (bool, optional, defaults to False) — Flag that will be used by default in the
                generate method of the model. Whether to stop the beam search when at least num_beams sentences are
                finished per batch or not.

length_penalty (float, optional, defaults to 1.0) — Exponential penalty to the length. 1.0 means no penalty. 
                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
                order to encourage the model to produce longer sequences.

num_beams (int, optional, defaults to 1) — Number of beams for beam search that will be used by default in the
                generate method of the model. 1 means no beam search.

remove_invalid_values (bool, optional) — Whether to remove possible nan and inf outputs of the model to prevent
                the generation method to crash. Note that using remove_invalid_values can slow down generation.

repetition_penalty (float, optional, defaults to 1) — Parameter for repetition penalty that will be used by default
                in the generate method of the model. 1.0 means no penalty.

length_penalty (float, optional, defaults to 1) — Exponential penalty to the length that will be used by default in
                the generate method of the model.
'''
model.config.vocab_size = model.config.encoder.vocab_size
model.config.max_length = 16
model.config.min_length = 4
model.config.no_repeat_ngram_size = 2
model.config.early_stopping = True
model.config.length_penalty = 1
model.config.num_beams = 4
# model.config.repetition_penalty = 20.9

# Save and Load Functions

def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    xm.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

def save_metrics(save_path,
                train_loss_list,
                valid_loss_list,
                global_steps_list,
                rouge_1_precision_list,
                rouge_1_recall_list,
                rouge_1_f1_list,
                rouge_2_precision_list,
                rouge_2_recall_list,
                rouge_2_f1_list,
                rouge_L_precision_list,
                rouge_L_recall_list,
                rouge_L_f1_list,
                bleu_score_list,
                bleu_precision_1_list,
                bleu_precision_2_list,
                bleu_precision_3_list,
                bleu_precision_4_list,
                brevity_penalty_list,
                length_ratio_list,
                translation_length_list,
                reference_length_list,
                bert_precision_list,
                bert_recall_list,
                bert_f1_list,
                meteor_score_list):
    if save_path == None:
        return
    
    state_dict = {
                'train_loss_list': train_loss_list,
                'valid_loss_list': valid_loss_list,
                'global_steps_list': global_steps_list,
                'rouge_1_precision_list': rouge_1_precision_list,
                'rouge_1_recall_list': rouge_1_recall_list,
                'rouge_1_f1_list': rouge_1_f1_list,
                'rouge_2_precision_list':rouge_2_precision_list,
                'rouge_2_recall_list':rouge_2_recall_list,
                'rouge_2_f1_list':rouge_2_f1_list,
                'rouge_L_precision_list':rouge_L_precision_list,
                'rouge_L_recall_list':rouge_L_recall_list,
                'rouge_L_f1_list':rouge_L_f1_list,
                'bleu_score_list':bleu_score_list,
                'bleu_precision_1_list':bleu_precision_1_list,
                'bleu_precision_2_list':bleu_precision_2_list,
                'bleu_precision_3_list':bleu_precision_3_list,
                'bleu_precision_4_list':bleu_precision_4_list,
                'brevity_penalty_list':brevity_penalty_list,
                'length_ratio_list':length_ratio_list,
                'translation_length_list':translation_length_list,
                'reference_length_list':reference_length_list,
                'bert_precision_list':bert_precision_list,
                'bert_recall_list':bert_recall_list,
                'bert_f1_list':bert_f1_list,
                'meteor_score_list':meteor_score_list
                }
    
    xm.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_metrics(load_path):
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return (state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list'],
            state_dict['global_steps_list'], state_dict['rouge_1_f1_list'], state_dict['rouge_2_precision_list'], 
            state_dict['rouge_2_recall_list'], state_dict['rouge_2_f1_list'], state_dict['rouge_L_precision_list'], 
            state_dict['rouge_L_recall_list'], state_dict['rouge_L_f1_list'], state_dict['bleu_score_list'], 
            state_dict['bleu_precision_1_list'], state_dict['bleu_precision_2_list'], state_dict['bleu_precision_3_list'], 
            state_dict['bleu_precision_4_list'], state_dict['brevity_penalty_list'], state_dict['length_ratio_list'], 
            state_dict['translation_length_list'], state_dict['reference_length_list'], state_dict['bert_precision_list'], 
            state_dict['bert_recall_list'], state_dict['bert_f1_list'], state_dict['meteor_score_list'], )

 # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.compute_metrics
bertscore = evaluate.load('bertscore')
bleu = datasets.load_metric("bleu")
meteor = evaluate.load('meteor')

rouge_scores_master_list = []

def getAverage(score_list):
    return sum(score_list) / len(score_list)

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

def bleu_score(pred_str, label_str):
    predictions = [tokenizer.tokenize(sentence) for sentence in pred_str]
    references = [[tokenizer.tokenize(sentence)] for sentence in label_str]
    bleu_score = bleu.compute(predictions=predictions, references=references)
    return bleu_score

def bert_score(pred_str, label_str):
    bertscore_output = bertscore.compute(predictions=pred_str, references=label_str, lang="bn")
    precision = getAverage(bertscore_output['precision'])
    recall = getAverage(bertscore_output['recall'])
    f1 = getAverage(bertscore_output['f1'])
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def meteor_score(pred_str, label_str):
    meteor_output = meteor.compute(predictions=pred_str, references=label_str)
    return meteor_output

'''
compute_metrics (Callable[[EvalPrediction], Dict], optional) — The function that will be used to compute metrics at
                evaluation. Must take a EvalPrediction and return a dictionary string to metric values.
'''
def compute_metrics(pred_ids, label_ids):
    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids==-100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # print(f"***pred_str: {pred_str},\n\tlabel_str: {label_str}\n")
    with open('base-model/evaluation_results_48_PA.txt', 'a') as f:
        f.write(f"***pred_str: {pred_str},\n\n\tlabel_str: {label_str}\n\n\n\n\n")

    rouge_output = rouge_score(pred_str, label_str)
    try:
        bleu_output = bleu_score(pred_str, label_str)
        bleu_output["precision_1_gram"] = bleu_output["precisions"][0]
        bleu_output["precision_2_gram"] = bleu_output["precisions"][1]
        bleu_output["precision_3_gram"] = bleu_output["precisions"][2]
        bleu_output["precision_4_gram"] = bleu_output["precisions"][3]
    except:
        bleu_output = {
            "bleu": 0.0,
            "precision_1_gram": 0.0,
            "precision_2_gram": 0.0,
            "precision_3_gram": 0.0,
            "precision_4_gram": 0.0,
            "brevity_penalty": 0.0,
            "length_ratio": 0.0,
            "translation_length": 0,
            "reference_length": 0
        }
    bertscore_output = bert_score(pred_str, label_str)
    meteor_output = meteor_score(pred_str, label_str)

    return {
        "rouge1_precision": round(rouge_output["rouge1"].precision, 4),
        "rouge1_recall": round(rouge_output["rouge1"].recall, 4),
        "rouge1_fmeasure": round(rouge_output["rouge1"].fmeasure, 4),
        "rouge2_precision": round(rouge_output["rouge2"].precision, 4),
        "rouge2_recall": round(rouge_output["rouge2"].recall, 4),
        "rouge2_fmeasure": round(rouge_output["rouge2"].fmeasure, 4),
        "rougeL_precision": round(rouge_output["rougeL"].precision, 4),
        "rougeL_recall": round(rouge_output["rougeL"].recall, 4),
        "rougeL_fmeasure": round(rouge_output["rougeL"].fmeasure, 4),
        "bleu_score": round(bleu_output["bleu"], 4),
        "bleu_precision_1": round(bleu_output["precision_1_gram"], 4),
        "bleu_precision_2": round(bleu_output["precision_2_gram"], 4),
        "bleu_precision_3": round(bleu_output["precision_3_gram"], 4),
        "bleu_precision_4": round(bleu_output["precision_4_gram"], 4),
        "brevity_penalty": round(bleu_output["brevity_penalty"], 4),
        "length_ratio": round(bleu_output["length_ratio"], 4),
        "translation_length": bleu_output["translation_length"],
        "reference_length": bleu_output["reference_length"],
        "bert_precision": round(bertscore_output["precision"], 4),
        "bert_recall": round(bertscore_output["recall"], 4),
        "bert_f1": round(bertscore_output["f1"], 4),
        "meteor_score": round(meteor_output['meteor'], 4)
    }

def train(model,
          optimizer,
          train_dataloader = train_dataloader,
          val_dataloader = val_dataloader,
          num_epochs = 5,
          eval_every = len(train_dataloader) // 2,
          file_path = destination_folder,
          best_valid_loss = float("Inf")):
    # initialize running values
    global_step = 0
    running_loss = 0.0
    valid_running_loss = 0.0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    for epoch in range(num_epochs):
        # initializing evaluation scores
        rouge_1_precision = 0.0
        rouge_1_recall = 0.0
        rouge_1_f1 = 0.0
        rouge_2_precision = 0.0
        rouge_2_recall = 0.0
        rouge_2_f1 = 0.0
        rouge_L_precision = 0.0
        rouge_L_recall = 0.0
        rouge_L_f1 = 0.0
        bleu_score = 0.0
        bleu_precision_1 = 0.0
        bleu_precision_2 = 0.0
        bleu_precision_3 = 0.0
        bleu_precision_4 = 0.0
        brevity_penalty = 0.0
        length_ratio = 0.0
        translation_length = 0
        reference_length = 0
        bert_precision = 0.0
        bert_recall = 0.0
        bert_f1 = 0.0
        meteor_score = 0.0
        rouge_1_precision_list = []
        rouge_1_recall_list = []
        rouge_1_f1_list = []
        rouge_2_precision_list = []
        rouge_2_recall_list = []
        rouge_2_f1_list = []
        rouge_L_precision_list = []
        rouge_L_recall_list = []
        rouge_L_f1_list = []
        bleu_score_list = []
        bleu_precision_1_list = []
        bleu_precision_2_list = []
        bleu_precision_3_list = []
        bleu_precision_4_list = []
        brevity_penalty_list = []
        length_ratio_list = []
        translation_length_list = []
        reference_length_list = []
        bert_precision_list = []
        bert_recall_list = []
        bert_f1_list = []
        meteor_score_list = []

        # train + evaluate on training data
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
            loss.backward()
            xm.optimizer_step(optimizer, barrier=True)

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step (batch generation)
            if global_step % eval_every == 0:
                model.eval()

                with torch.no_grad():
                    for batch in tqdm(val_dataloader):
                        val_input_ids = batch["input_ids"].to(device)
                        val_attention_mask = batch["attention_mask"].to(device)
                        val_labels = batch["labels"].to(device)

                        val_outputs = model(input_ids=val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
                        val_loss = val_outputs.loss
                        valid_running_loss += val_loss.item()

                        # generating prediction for evaluation metrics
                        pred_ids = model.generate(batch["input_ids"].to(device))
                        # compute metrics
                        metrics = compute_metrics(pred_ids=pred_ids, label_ids=batch["labels"])
                        # rouge score
                        rouge_1_precision += metrics["rouge1_precision"] 
                        rouge_1_recall += metrics["rouge1_recall"] 
                        rouge_1_f1 += metrics["rouge1_fmeasure"] 
                        rouge_2_precision += metrics["rouge2_precision"] 
                        rouge_2_recall += metrics["rouge2_recall"] 
                        rouge_2_f1 += metrics["rouge2_fmeasure"] 
                        rouge_L_precision += metrics["rougeL_precision"] 
                        rouge_L_recall += metrics["rougeL_recall"] 
                        rouge_L_f1 += metrics["rougeL_fmeasure"]
                        # bleu score
                        bleu_score += metrics["bleu_score"]
                        bleu_precision_1 += metrics["bleu_precision_1"]
                        bleu_precision_2 += metrics["bleu_precision_2"]
                        bleu_precision_3 += metrics["bleu_precision_3"]
                        bleu_precision_4 += metrics["bleu_precision_4"]
                        brevity_penalty += metrics["brevity_penalty"]
                        length_ratio += metrics["length_ratio"]
                        translation_length += metrics["translation_length"]
                        reference_length += metrics["reference_length"]
                        # bert score
                        bert_precision += metrics["bert_precision"]
                        bert_recall += metrics["bert_recall"]
                        bert_f1 += metrics["bert_f1"]
                        # meteor score
                        meteor_score += metrics["meteor_score"]


                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(val_dataloader)
                # rouge
                average_rouge_1_precision = rouge_1_precision / len(val_dataloader)
                average_rouge_1_recall = rouge_1_recall / len(val_dataloader)
                average_rouge_1_f1 = rouge_1_f1 / len(val_dataloader)
                average_rouge_2_precision = rouge_2_precision / len(val_dataloader)
                average_rouge_2_recall = rouge_2_recall / len(val_dataloader)
                average_rouge_2_f1 = rouge_2_f1 / len(val_dataloader)
                average_rouge_L_precision = rouge_L_precision / len(val_dataloader)
                average_rouge_L_recall = rouge_L_recall / len(val_dataloader)
                average_rouge_L_f1 = rouge_L_f1 / len(val_dataloader)
                # bleu
                average_bleu_score = bleu_score / len(val_dataloader)
                average_bleu_precision_1 = bleu_precision_1 / len(val_dataloader)
                average_bleu_precision_2 = bleu_precision_2 / len(val_dataloader)
                average_bleu_precision_3 = bleu_precision_3 / len(val_dataloader)
                average_bleu_precision_4 = bleu_precision_4 / len(val_dataloader)
                average_brevity_penalty = brevity_penalty / len(val_dataloader)
                average_length_ratio = length_ratio / len(val_dataloader)
                average_translation_length = translation_length / len(val_dataloader)
                average_reference_length = reference_length / len(val_dataloader)
                # bert
                average_bert_precision = bert_precision / len(val_dataloader)
                average_bert_recall = bert_recall / len(val_dataloader)
                average_bert_f1 = bert_f1 / len(val_dataloader)
                # meteor
                average_meteor_score = meteor_score / len(val_dataloader)

                # append values to the master list
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)
                # rouge
                rouge_1_precision_list.append(average_rouge_1_precision)
                rouge_1_recall_list.append(average_rouge_1_recall)
                rouge_1_f1_list.append(average_rouge_1_f1)
                rouge_2_precision_list.append(average_rouge_2_precision)
                rouge_2_recall_list.append(average_rouge_2_recall)
                rouge_2_f1_list.append(average_rouge_2_f1)
                rouge_L_precision_list.append(average_rouge_L_precision)
                rouge_L_recall_list.append(average_rouge_L_recall)
                rouge_L_f1_list.append(average_rouge_L_f1)
                # bleu
                bleu_score_list.append(average_bleu_score)
                bleu_precision_1_list.append(average_bleu_precision_1)
                bleu_precision_2_list.append(average_bleu_precision_2)
                bleu_precision_3_list.append(average_bleu_precision_3)
                bleu_precision_4_list.append(average_bleu_precision_4)
                brevity_penalty_list.append(average_brevity_penalty)
                length_ratio_list.append(average_length_ratio)
                translation_length_list.append(average_translation_length)
                reference_length_list.append(average_reference_length)
                # bert
                bert_precision_list.append(average_bert_precision)
                bert_recall_list.append(average_bert_recall)
                bert_f1_list.append(average_bert_f1)
                # meteor
                meteor_score_list.append(average_meteor_score)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                    .format(epoch+1, num_epochs, global_step, num_epochs*len(train_dataloader),
                            average_train_loss, average_valid_loss))
                
                # rouge score
                print("\tROUGE 1:")
                print("\t\tprecision:", average_rouge_1_precision)
                print("\t\trecall:", average_rouge_1_recall)
                print("\t\tf1:", average_rouge_1_f1)
                print("\tROUGE 2:")
                print("\t\tprecision:", average_rouge_2_precision)
                print("\t\trecall:", average_rouge_2_recall)
                print("\t\tf1:", average_rouge_2_f1)
                print("\tROUGE L:")
                print("\t\tprecision:", average_rouge_L_precision)
                print("\t\trecall:", average_rouge_L_recall)
                print("\t\tf1:", average_rouge_L_f1)
                # bleu score
                print("\tBleu score: ", average_bleu_score)
                print("\tBleu precision 1: ", average_bleu_precision_1)
                print("\tBleu precision 2: ", average_bleu_precision_2)
                print("\tBleu precision 3: ", average_bleu_precision_3)
                print("\tBleu precision 4: ", average_bleu_precision_4)
                print("\tBrevity penalty: ", average_brevity_penalty)
                print("\tLength ratio: ", average_length_ratio)
                print("\tTranslation length: ", average_translation_length)
                print("\tReference length: ", average_reference_length)
                # bert score
                print("\tBERT:")
                print("\t\tprecision: ", average_bert_precision)
                print("\t\trecall: ", average_bert_recall)
                print("\t\tf1: ", average_bert_f1)
                # meteor score
                print("\tMeteor score: ", average_meteor_score)
                    
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'best_model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'best_metrics.pt',
                                train_loss_list,
                                valid_loss_list,
                                global_steps_list,
                                rouge_1_precision_list,
                                rouge_1_recall_list,
                                rouge_1_f1_list,
                                rouge_2_precision_list,
                                rouge_2_recall_list,
                                rouge_2_f1_list,
                                rouge_L_precision_list,
                                rouge_L_recall_list,
                                rouge_L_f1_list,
                                bleu_score_list,
                                bleu_precision_1_list,
                                bleu_precision_2_list,
                                bleu_precision_3_list,
                                bleu_precision_4_list,
                                brevity_penalty_list,
                                length_ratio_list,
                                translation_length_list,
                                reference_length_list,
                                bert_precision_list,
                                bert_recall_list,
                                bert_f1_list,
                                meteor_score_list)
    

    save_metrics(file_path + '/' + 'metrics.pt',
                train_loss_list,
                valid_loss_list,
                global_steps_list,
                rouge_1_precision_list,
                rouge_1_recall_list,
                rouge_1_f1_list,
                rouge_2_precision_list,
                rouge_2_recall_list,
                rouge_2_f1_list,
                rouge_L_precision_list,
                rouge_L_recall_list,
                rouge_L_f1_list,
                bleu_score_list,
                bleu_precision_1_list,
                bleu_precision_2_list,
                bleu_precision_3_list,
                bleu_precision_4_list,
                brevity_penalty_list,
                length_ratio_list,
                translation_length_list,
                reference_length_list,
                bert_precision_list,
                bert_recall_list,
                bert_f1_list,
                meteor_score_list)
    print('Finished Training!')

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = xm.xla_device()
model.to(device)

train(model=model, optimizer=optimizer, num_epochs=30)