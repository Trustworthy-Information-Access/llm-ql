# LLM-QL 
This repository contains the code, datasets models used in our paper: "Unleashing the Power of LLMs in Dense Retrieval with Query
Likelihood Modeling" 

Dense retrieval is a crucial task in Information Retrieval (IR), serving as the basis for downstream tasks such as re-ranking and augmenting generation. Recently, large language models (LLMs) have demonstrated impressive semantic understanding capabilities, making them attractive to researchers focusing on dense retrieval. While LLMs, as decoder-style generative models, excel in language generation, they often fall short in modeling global information due to a lack of attention to subsequent tokens. Drawing inspiration from the classical word-based language modeling approach for IR, specifically the query likelihood (QL) model, we aim to leverage the generative strengths of LLMs through QL maximization. Rather than employing QL estimation for document ranking, we propose an auxiliary task of QL maximization to enhance the backbone for subsequent contrastive learning of the retriever. We introduce our model, LLM-QL, which incorporates two key components: Attention Block (AB) and Document Corruption (DC). AB blocks the attention of predictive tokens to the document tokens before the document's ending token, while DC corrupts a document by masking a portion of its tokens during prediction. Evaluations on the in-domain (MS MARCO) and out-of-domain dataset (BEIR) indicate LLM-QL's superiority over other LLM-based retrievers. Furthermore, comprehensive analyses also validate the efficacy of LLM-QL and its components.


# Download dataset 
We use the [MSMARCO v1 and TREC-DL](https://microsoft.github.io/msmarco/Datasets) and  [BEIR](https://github.com/beir-cellar/beir).  
We use the hard negative samples provided by [Tevatron](https://www.dropbox.com/scl/fi/pkm1mtgfobae9kuesp7dr/train-tevatron.jsonl?rlkey=2thutc4zkozr9jp4zbbrz5rvi&dl=0). 


# Run
## QL Modeling
```
Deepspeed --num_gpus=8 ql-learning/run.py \
  --deepspeed ql-learning/ds_zero3_config.json \
  --output_dir $encode_path1 \
  --model_name_or_path $ori_model_path1 \
  --train_data $training_path \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --rand_num 0.1 \
  --mask_probability 0.60 \
  --cut_lens 0 \
  --q_max_len 200 \
  --p_max_len 200 \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --dataloader_drop_last True \
  --cutoff_len 512 \
  --logging_steps 1 \
  --save_steps 200 \
  --save_total_limit 20 \
  --gradient_checkpointing \
  --ddp_find_unused_parameters False \
  --use_flash_attn True \
  --warmup_ratio 0.1 \
  --remove_stop_words False \
  --use_lora False \
  --bf16 True \
```

## Contrastive Learning
```
cd contrastive_learning/llm-index/tevatron-main/src
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 60001 --module tevatron.retriever.driver.train \
  --deepspeed ../../../ql-learning/ds_zero3_config.json \
  --output_dir  $lora_model_save_path \
  --model_name_or_path $model_path \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 200 \
  --lora_r 32 \
  --dataset_path $train_path \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 4 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len $q_max_len \
  --passage_max_len $p_max_len \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --gradient_accumulation_steps 4
```
