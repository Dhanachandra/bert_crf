#!/usr/bin/env python
# coding: utf-8
__author__ = 'Dhanachandra N.'

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle 
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils import data 
from transformers import BertTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import os
from transformers import BertPreTrainedModel, BertModel
from torchcrf import CRF
import timeit
import subprocess
from tqdm import tqdm, trange
from transformers import AdamW, WarmupLinearSchedule
from matplotlib import pyplot as plt 
import datetime
from param import PARAM
import spacy
tokenizer = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])

log_soft = F.log_softmax

#to initialize the network weight with fix seed. 
def seed_torch(seed=12345):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()

from collections import OrderedDict
# read the corpus and return them into list of sentences of list of tokens
def corpus_reader(path, delim='\t', word_idx=0, beg_idx=1, end_idx=2, fname_idx=3, label_idx=-1):
    tokens, begins, ends, file_names, labels = [], [], [], [], []
    tmp_tok, tmp_beg, tmp_end, tmp_fname, tmp_lab = [], [], [], [], []
    label_set = []
    with open(path, 'r') as reader:
        for line in reader:
            line = line.strip()
            cols = line.split(delim)
            if len(cols) < 2:
                if len(tmp_tok) > 0:
                    tokens.append(tmp_tok); begins.append(tmp_beg), ends.append(tmp_end), file_names.append(tmp_fname); labels.append(tmp_lab)
                tmp_tok = []
                tmp_beg = []
                tmp_end = []
                tmp_fname = []
                tmp_lab = []
            else:
                tmp_tok.append(cols[word_idx])
                tmp_beg.append(cols[beg_idx])
                tmp_end.append(cols[end_idx])
                tmp_fname.append(cols[fname_idx])
                tmp_lab.append(cols[label_idx])
                label_set.append(cols[label_idx])
  return tokens, begins, ends, file_names, labels, list(OrderedDict.fromkeys(label_set))

class NER_Dataset(data.Dataset):
    def __init__(self, sentences, begins, ends, fnames, labels, tokenizer_path = '', do_lower_case=True):
       self.sentences = sentences
       self.begins = begins
       self.ends = ends
       self.fnames = fnames
       self.labels = labels
       self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=do_lower_case)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        label = []
        for x in self.labels[idx]:
            if x in tag2idx.keys():
                label.append(tag2idx[x])
            else:
                label.append(tag2idx['O'])
        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append('[CLS]')
        #append dummy label 'X' for subtokens
        modified_labels = [tag2idx['X']]
        for i, token in enumerate(sentence):
            if len(bert_tokens) >= 512:
                break
            orig_to_tok_map.append(len(bert_tokens))
            modified_labels.append(label[i])
            new_token = self.tokenizer.tokenize(token)
            bert_tokens.extend(new_token)
            modified_labels.extend([tag2idx['X']] * (len(new_token) -1))

        bert_tokens.append('[SEP]')
        modified_labels.append(tag2idx['X'])
        token_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        f_name = self.fnames[idx][0]
        if len(token_ids) > 511:
            token_ids = token_ids[:512]
            modified_labels = modified_labels[:512]

        return token_ids, len(token_ids), orig_to_tok_map, modified_labels, self.begins[idx], self.ends[idx],  self.sentences[idx], self.fnames[idx]
 
def pad(batch):
    '''Pads to the longest sample'''
    get_element = lambda x: [sample[x] for sample in batch]
    seq_len = get_element(1)
    maxlen = np.array(seq_len).max()
    do_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    tok_ids = do_pad(0, maxlen)
    attn_mask = [[(i>0) for i in ids] for ids in tok_ids] 
    LT = torch.LongTensor
    label = do_pad(3, maxlen)
    
    # sort the index, attn mask and labels on token length
    token_ids = get_element(0)
    token_ids_len = torch.LongTensor(list(map(len, token_ids)))
    _, sorted_idx = token_ids_len.sort(0, descending=True)

    tok_ids = LT(tok_ids)[sorted_idx]
    attn_mask = LT(attn_mask)[sorted_idx]
    labels = LT(label)[sorted_idx]

    orig_tok_map = get_element(2)
    begins = get_element(4)
    ends = get_element(5)
    sents = get_element(6)
    f_names = get_element(7)

    return tok_ids, attn_mask, org_tok_map, labels, begins, ends, sents, f_names, list(sorted_idx.cpu().numpy())

class Bert_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_CRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        self.crf = CRF(self.num_labels, batch_first=True)    
    
    def forward(self, input_ids, attn_masks, labels=None):  # dont confuse this with _forward_alg above.
        outputs = self.bert(input_ids, attn_masks)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output)        
        attn_masks = attn_masks.type(torch.uint8)
        if labels is not None:
            loss = -self.crf(log_soft(emission, 2), labels, mask=attn_masks, reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emission, mask=attn_masks)
            return prediction

def generate_training_data(param, bert_tokenizer="bert-base", do_lower_case=True):
    training_data, validation_data = param.apr_dir+param.training_data, param.apr_dir+param.val_data 
    train_sentences, begins, ends, file_names, train_labels, tag2idx = corpus_reader(training_data, delim='\t')
    train_dataset = NER_Dataset(train_sentences, begins, ends, file_names, train_labels, tokenizer_path = bert_tokenizer, do_lower_case=do_lower_case)
    # save the tag2indx dictionary. Will be used while prediction
    with open(APR_DIR + 'tag2idx.pkl', 'wb') as f:
        pickle.dump(tag2idx, f, pickle.HIGHEST_PROTOCOL)
    dev_sentences, dev_begins, dev_ends, dev_file_names, dev_labels, _ = corpus_reader(validation_data, delim='\t')
    dev_dataset = NER_Dataset(dev_sentences, dev_begins, dev_ends, dev_file_names, dev_labels, tokenizer_path = bert_tokenizer, do_lower_case=do_lower_case)

    train_iter = data.DataLoader(dataset=train_dataset,
                                batch_size=16,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=pad)
    eval_iter = data.DataLoader(dataset=dev_dataset,
                                batch_size=16,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)
    return train_iter, eval_iter, tag2idx

def generate_test_data(param, bert_tokenizer="bert-base", do_lower_case=True):
    test_data = param.apr_dir+param.test_data
    test_sentences, test_begins, test_ends, test_file_names, test_labels, _ = corpus_reader(test_data, delim='\t')
    test_dataset = NER_Dataset(test_sentences, test_begins, test_ends, test_file_names, test_labels, tokenizer_path = bert_tokenizer, do_lower_case=do_lower_case)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=16,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)
    return test_iter


def train(train_iter, eval_iter, tag2idx, param, bert_model="bert-base"):
    print('#Tags: ', len(tag2idx))
    unique_labels = list(tag2idx.keys())
    model = Bert_CRF.from_pretrained("bert-base", num_labels = len(tag2idx))
    if torch.cuda.is_available():
      model.cuda()
    num_epoch = param.epoch
    gradient_acc_steps = 1
    t_total = len(train_iter) // gradient_acc_steps * num_epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=param.lr, eps=param.eps)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=t_total)
    global_step = 0
    model.zero_grad()
    model.train()
    training_loss = []
    validation_loss = []
    pearson_score = []
    train_iterator = trange(num_epoch, desc="Epoch", disable=0)
    best_f1 = 0.0
    tmp_loss = 0.0
    MAX_SCORE = 0.0 
    MAX_EPOCH = 0
    start_time = timeit.default_timer()

    for epoch in (train_iterator):
        epoch_iterator = tqdm(train_iter, desc="Iteration", disable=-1)
        tr_loss = 0.0
        model.train()
        for step, batch in enumerate(epoch_iterator):
            s = timeit.default_timer()
            token_ids, attn_mask, _, labels, _, _, _, _, _= batch
            #print(labels)
            inputs = {'input_ids' : token_ids.to(device),
                     'attn_masks' : attn_mask.to(device),
                     'labels' : labels.to(device)
                     }  
            loss= model(**inputs) 
            loss.backward()
            tmp_loss += loss.item()
            tr_loss += loss.item()
            if (step + 1) % 1 == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            if step == 0:
                print('%s Step: %d of %d Loss: %f' %(str(datetime.datetime.now()), (step+1), len(epoch_iterator), tmp_loss))
            if (step+1) % 1000 == 0:
                print('%s Step: %d of %d Loss: %f' %(str(datetime.datetime.now()), (step+1), len(epoch_iterator), tmp_loss/1000))
                tmp_loss = 0.0
      
        print("Training Loss: %f for epoch %d" %(tr_loss/len(train_iter), epoch))
        training_loss.append(tr_loss/len(train_iter))
        #'''
        #Y_pred = []
        #Y_true = []
        val_loss = 0.0
        model.eval()
        writer = open(APR_DIR + 'prediction_'+str(epoch)+'.csv', 'w')
        for i, batch in enumerate(eval_iter):
            token_ids, attn_mask, org_tok_map, labels, begins, ends, original_token, f_names, sorted_idx = batch
            #attn_mask.dt
            inputs = {'input_ids': token_ids.to(device),
                      'attn_masks' : attn_mask.to(device)
                     }  
            
            dev_inputs = {'input_ids' : token_ids.to(device),
                         'attn_masks' : attn_mask.to(device),
                         'labels' : labels.to(device)
                         } 
            with torch.torch.no_grad():
                tag_seqs = model(**inputs)
                tmp_eval_loss = model(**dev_inputs)
            val_loss += tmp_eval_loss.item()
            #print(labels.numpy())
            y_true = list(labels.cpu().numpy())
            for i in range(sorted_idx):
                o2m = org_tok_map[i]
                pos = sorted_idx.index(i)
                for j, orig_tok_idx in enumerate(o2m):
                    writer.write(original_token[i][j] + '\t')
                    writer.write(begins[i][j] + '\t')
                    writer.write(ends[i][j] + '\t')
                    writer.write(f_names[i][j] + '\t')
                    writer.write('SCORE\t')
                    writer.write(unique_labels[y_true[pos][orig_tok_idx]] + '\t')
                    pred_tag = unique_labels[tag_seqs[pos][orig_tok_idx]]
                    if pred_tag == 'X':
                        pred_tag = 'O'
                    writer.write(pred_tag + '\n')
                writer.write('\n')
                
        validation_loss.append(val_loss/len(eval_iter))
        writer.flush()
        print('Epoch: ', epoch)
        command = "python conlleval.py < " + APR_DIR + "prediction_"+str(epoch)+".csv"
        process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
        result = process.communicate()[0].decode("utf-8")
        print(result)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss/len(train_iter),
        }, APR_DIR + 'model_' + str(epoch) + '.pt')

    total_time = timeit.default_timer() - start_time
    print('Total training time: ',   total_time)
    return training_loss, validation_loss

'''
    raw_text should pad data in raw data prediction
'''
def test(test_iter, model, unique_labels, test_output):
    writer = open(APR_DIR + test_output, 'w')
    for i, batch in enumerate(eval_iter):
        token_ids, attn_mask, org_tok_map, labels, begins, ends, original_token, f_names, sorted_idx = batch
        #attn_mask.dt
        inputs = {'input_ids': token_ids.to(device),
                  'attn_masks' : attn_mask.to(device)
                 }  
        with torch.torch.no_grad():
            tag_seqs = model(**inputs)
        y_true = list(labels.cpu().numpy())
        for i in range(sorted_idx):
            o2m = org_tok_map[i]
            pos = sorted_idx.index(i)
            for j, orig_tok_idx in enumerate(o2m):
                writer.write(original_token[i][j] + '\t')
                writer.write(begins[i][j] + '\t')
                writer.write(ends[i][j] + '\t')
                writer.write(filenames[i][j] + '\t')
                writer.write('SCORE\t')
                writer.write(unique_labels[y_true[pos][orig_tok_idx]] + '\t')
                pred_tag = unique_labels[tag_seqs[pos][orig_tok_idx]]
                if pred_tag == 'X':
                    pred_tag = 'O'
                writer.write(pred_tag + '\n')
            writer.write('\n')
    validation_loss.append(val_loss/len(eval_iter))
    writer.flush()
    print('Epoch: ', epoch)
    command = "python conlleval.py < " + APR_DIR + test_output
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    result = process.communicate()[0].decode("utf-8")

def parse_raw_data(padded_raw_data, model, unique_labels, out_file_name='raw_prediction.csv'):
    token_ids, attn_mask, org_tok_map, labels, begins, ends, original_token, f_names, sorted_idx = padded_raw_data
    #attn_mask.dt
    writer = open(out_file_name, 'w')
    inputs = {'input_ids': token_ids.to(device),
              'attn_masks' : attn_mask.to(device)
             }  
    with torch.torch.no_grad():
        tag_seqs = model(**inputs)
    y_true = list(labels.cpu().numpy())
    for i in range(sorted_idx):
        o2m = org_tok_map[i]
        pos = sorted_idx.index(i)
        for j, orig_tok_idx in enumerate(o2m):
            writer.write(original_token[i][j] + '\t')
            writer.write(begins[i][j] + '\t')
            writer.write(ends[i][j] + '\t')
            writer.write(filenames[i][j] + '\t')
            writer.write('SCORE\t')
            writer.write(unique_labels[y_true[pos][orig_tok_idx]] + '\t')
            pred_tag = unique_labels[tag_seqs[pos][orig_tok_idx]]
            if pred_tag == 'X':
                pred_tag = 'O'
            writer.write(pred_tag + '\n')
        writer.write('\n')
    print("Raw data prediction done!")

def show_graph(training_loss, validation_loss, resource_dir):
    plt.plot(range(1,17), training_loss, label='Training Loss')
    plt.plot(range(1,17), validation_loss, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Training Loss Vs Testing Loss")
    plt.legend()
    plt.show()
    plt.savefig(resource_dir + 'Loss.png')

def load_model(param, do_lower_case=True):
    f = open(param.apr_dir +'tag2idx.pkl', 'rb')
    tag2idx = pickle.load(f)
    unique_labels = list(tag2idx.keys())
    model = Bert_CRF.from_pretrained(param.apr_dir+param.model_name, num_labels=len(tag2idx))
    checkpoint = torch.load(ner_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    global bert_tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(ezbert_path, do_lower_case=do_lower_case)
    if torch.cuda.is_available():
        model.cuda()
    logger.debug('return from model function')
    toc = time.time()
    model.eval()
    return model, bert_tokenizer, unique_labels

def raw_processing(doc, bert_tokenizer, word_tokenizer):
    tic = time.time()
    spans = re.split("[\n\r]", doc)
    offset = 0
    batch = []
    for span in spans:
        sentences = sentence_segmenter(span)
        for s_idx, sentence in enumerate(sentences.sents):
            bert_tokens = []
            orig_to_tok_map = []
            bert_tokens.append('[CLS]')
            begins = []
            ends = []
            for tok in tokenzer(word):
            	token = tok.text
                offset = doc.find(token, offset)
                current_begins.append(offset)
                ends.append(offset + len(token))
                offset += len(token)
                orig_to_tok_map.append(len(bert_tokens))
                new_token = bert_tokenizer.tokenize(token)
                bert_tokens.extend(new_token)
            bert_tokens.append('[SEP]')
            token_id = bert_tokenizer.convert_tokens_to_ids(bert_tokens)
            if len(token_id) > 511:
                token_id = token_id[:512]
            dummy_labels = ['X'] * len(token_id)
            dummy_f_names = ['f_names'] * len(token_id)
            sample = (token_id, len(token_id), orig_to_tok_map, dummy_labels, begins, ends, original_token, dummy_f_names)
            batch.append(sample)
    pad_data = pad(batch)
    return pa_data    

if __name__ == "__main__":
    if PARAM.mode == "train":
        train_iter, eval_iter, tag2idx = generate_training_data(param=PARAM, bert_tokenizer=PARAM.bert_model, do_lower_case=True)
        t_loss, v_loss = train(train_iter, eval_iter, tag2idx, param=PARAM, bert_model=PARAM.bert_model)
        show_graph(t_loss, v_loss, PARAM.apr_dir)
    elif PARAM.mode = "prediction":
        test_iter = generate_test_data(param=PARAM, bert_tokenizer=PARAM.bert_model, do_lower_case=True)
        model, bert_tokenizer, unique_labels = load_model(param=PARAM, do_lower_case=True)
        test(test_iter, model, unique_labels, PARAM.test_out):
    elif PARAM.mode = "raw_text":
        if PARAM.raw_text == None:
            print('Please provide the raw text path on PARAM.raw_text')
            import sys
            sys.exit(1)
        model, bert_tokenizer, unique_labels = load_model(param=PARAM, do_lower_case=True)
        doc = open(PARAM.raw_text).read()
        pad_data = raw_processing(doc, bert_tokenizer)
        parse_raw_data(pad_data, model, unique_labels, out_file_name=PARAM.raw_prediction_output)
