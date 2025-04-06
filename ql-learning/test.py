import torch
import random
def format_query(query: str, prefix: str = '') -> str:
    return f'{prefix} {query.strip()}'.strip()

def format_passage(text: str, title: str = '', prefix: str = '') -> str:
    return f'{prefix} {title.strip()} {text.strip()}'.strip()

dataset = [{"query_id": "706678", "query": "what is a yowie", "positive_passages": [{"docid": "8841643", "title": "Thread: whats the difference between a sasquatch/yeti/bigfoot??", "text": "Yowie is one of several names given to a hominid reputed to live in the Australian wilderness. The creature has its roots in Aboriginal oral history."}]},
{"query_id": "405466", "query": "is carbonic acid soluble", "positive_passages": [{"docid": "8841735", "title": "Is carbonic acid (H2CO3) soluble?", "text": "Carbonic acid is unstable and decomposes to CO2 and H2O like titanium007 said. The foaming is evolution of CO2. Your equation is not completely right. The end products are CaCl2 and CO2. Carbon dioxide then reacts with water which is present to form carbonic acid, H2CO3.Excess CO2 escapes as you saw. in an acid-base reaction where the base is a carbonate (CO3 or bicarbonate HCO3) three products will form: water, carbon dioxide and a salt. the carbon dioxide is what you obsered with the fizzing.CaCl2 you have correctly identified as the salt.nd yes, carbonic acid is soluble. it's dissolved in soda, hence the carbonation when you open the bottle and releases the pressure. More Questions and Answers: The answers post by the user, for information only, FunQA.com does not guarantee the right."}]}]
from transformers import AutoTokenizer, HfArgumentParser, set_seed, AutoConfig, Trainer
from modeling import PreLlamaModel
def get_dataset(dataset):
    dataset =dataset[:1]
    tokenizer = AutoTokenizer.from_pretrained(
        '/root/paddlejob/workspace/env_run/model/llama2-7b',
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    des = ", retrieve relevant passages that answer the query: "
    des_ids = tokenizer(des, truncation=True, return_tensors=None, add_special_tokens=False)['input_ids']
    s_id = tokenizer('</s>', truncation=True, return_tensors=None, add_special_tokens=False)['input_ids']
    input_idses = []
    attention_masks = []
    labelss = []
    for item in range(len(dataset)):
        s_id_encoded = tokenizer('</s>', truncation=True, return_tensors=None, add_special_tokens=False)['input_ids']
        query = "<s>Instruct: Given a web search query, retrieve the most relevant passage that answer the query.\nQuery: " + dataset[item]['query'] + "\nThe most relevant passage: "
        positive_passages = dataset[item]['positive_passages']
        tokenizer.padding_side = "left" 
        query_id_encoded = tokenizer(query, truncation=True, max_length=100, padding='max_length', return_tensors=None, add_special_tokens=False)
        passage = positive_passages[random.randint(0, len(positive_passages)-1)]
        passages_all = format_passage(passage['text'], passage['title'])+"</s>"
        passages_encoded = tokenizer(passages_all, truncation=True, max_length=100, return_tensors=None, add_special_tokens=False)
        input_ids = query_id_encoded['input_ids'] + s_id_encoded + passages_encoded['input_ids']
        attention_mask = query_id_encoded['attention_mask'] + [1] + passages_encoded['attention_mask']
        labels = [-100 if i ==0 else i for i in query_id_encoded['input_ids']] + s_id_encoded + passages_encoded['input_ids']
        print(input_ids)
        input_idses.append(input_ids)
        attention_masks.append(attention_mask)
        labelss.append(labels)
    return torch.tensor(input_idses).cpu(), torch.tensor(attention_masks).cpu(), torch.tensor(labelss).cpu()
input_ids, attention_mask, labels = get_dataset(dataset)
model = PreLlamaModel.from_pretrained('/root/paddlejob/workspace/env_run/model/llama2-7b').cpu()
pp = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
print(pp)