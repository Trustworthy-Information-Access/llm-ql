import os.path
import random
import sys
import time
from dataclasses import dataclass
import re


import datasets
import numpy as np
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, AutoTokenizer, DataCollatorForSeq2Seq, PreTrainedTokenizer

from arguments import DataArguments
def format_query(query: str, prefix: str = '') -> str:
    return f'{prefix} {query.strip()}'.strip()

def format_passage(text: str, title: str = '', prefix: str = '') -> str:
    return f'{prefix} {title.strip()} {text.strip()}'.strip()

def mask_tokens(text, mask_probability=0.45, mask_token="_"):
    words = text.split()  
    masked_words = []
    if mask_probability == 0:
        return text
    for word in words:
        if random.random() < mask_probability: 
            masked_words.append(mask_token)
        else:
            masked_words.append(word)
    return ' '.join(masked_words)
def cut_lens(text, lens=100):
    words = text.split() 
    if  lens == 0:
        masked_words = words
    else:
        masked_words = words[:lens]
    return ' '.join(masked_words)
class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            args: DataArguments,
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train',
                                                 cache_dir=args.cache_path)

        self.args = args
        self.total_len = len(self.dataset)
        self.q_max_len = args.q_max_len
        self.tokenizer = tokenizer
        self.s_id = self.tokenizer('</s>', truncation=True, return_tensors=None, add_special_tokens=False)['input_ids']
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        if self.args.rand_num <= 0:
            rand_num = random.random()
        else:
            rand_num = self.args.rand_num
        if rand_num > 0.5:#q-d
            s_id_encoded = self.s_id
            query = "<s>Instruct: Given a web search query, retrieve the most relevant passage that answer the query.\nQuery: " + self.dataset[item]['query'] + "\nThe most relevant passage: "
            positive_passages = self.dataset[item]['positive_passages']
            self.tokenizer.padding_side = "left" 
            query_id_encoded = self.tokenizer(query, truncation=True, max_length=self.q_max_len, padding='max_length', return_tensors=None, add_special_tokens=False)
            passage = positive_passages[random.randint(0, len(positive_passages)-1)]
            passages_all = cut_lens(format_passage(passage['text'], passage['title']), self.args.cut_lens)+"</s>"
            passages_encoded = self.tokenizer(passages_all, truncation=True, max_length=self.args.p_max_len, return_tensors=None, add_special_tokens=False)
            input_ids = query_id_encoded['input_ids'] + s_id_encoded + passages_encoded['input_ids']
            attention_mask = query_id_encoded['attention_mask'] + [1] + passages_encoded['attention_mask']
            labels = [-100]*len(query_id_encoded['input_ids']) + [-100]*len(s_id_encoded) + passages_encoded["input_ids"]
            # labels = [-100 if i ==0 else i for i in query_id_encoded['input_ids']] + s_id_encoded + passages_encoded['input_ids']
        else:#d-q
            positive_passages = self.dataset[item]['positive_passages']
            passage = positive_passages[random.randint(0, len(positive_passages)-1)]
            passages_all_original = format_passage(passage['text'], passage['title'])
            passages_all = mask_tokens(passages_all_original, self.args.mask_probability)
            document = "<s>Instruct: Given a retrieved passage, summarize the passage.\nPassage: " + passages_all + "\nSummarization: "
            self.tokenizer.padding_side = "left" 
            document_id_encoded = self.tokenizer(document, truncation=True, max_length=self.args.p_max_len, padding='max_length', return_tensors=None, add_special_tokens=False)
            s_id_encoded = self.s_id
            query = self.dataset[item]['query']+"</s>"
            query_encoded = self.tokenizer(query, truncation=True, max_length=self.args.q_max_len, return_tensors=None, add_special_tokens=False)
            input_ids = document_id_encoded['input_ids'] + s_id_encoded + query_encoded["input_ids"]
            attention_mask = document_id_encoded['attention_mask'] + [1] + query_encoded["attention_mask"]
            # labels = [-100]*len(document_id_encoded['input_ids']) + [-100]*len(s_id_encoded) + query_encoded["input_ids"]
            labels = [-100 if i ==0 else i for i in document_id_encoded['input_ids']] + s_id_encoded + query_encoded["input_ids"]
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            "labels": labels
        }
        return result
@dataclass
class EmbedCollator(DataCollatorForSeq2Seq):
    cutoff_len: int = 512
    def __call__(self, features, return_tensors='pt'):
        if return_tensors is None:
            return_tensors = self.return_tensors
        inputs = [feature for feature in features]
        labels = [feature["labels"] for feature in inputs] if "labels" in inputs[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )
            self.tokenizer.padding_side = "right"
            padding_side = self.tokenizer.padding_side
            for feature in inputs:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        inputs = self.tokenizer.pad(
            inputs,
            padding=self.padding,
            max_length=self.cutoff_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
            return_attention_mask=True,
        ) 
        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=inputs["labels"])
            inputs["decoder_input_ids"] = decoder_input_ids
        # print(inputs['input_ids'].tolist())
        # assert 1 > 2
        return {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "labels": inputs['labels']
        }
