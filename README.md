# LLM-QL 
This repository contains the code, datasets models used in our paper: "Unleashing the Power of LLMs in Dense Retrieval with Query
Likelihood Modeling" 

Dense retrieval is a crucial task in Information Retrieval (IR) and is the foundation for downstream tasks such as re-ranking. Recently, large language models (LLMs) have shown compelling semantic understanding capabilities and are appealing to researchers studying dense retrieval. LLMs, as decoder-style generative models, are competent at language generation while falling short on modeling global information due to the lack of attention to tokens afterward. Inspired by the classical word-based language modeling approach for IR, i.e., the query likelihood (QL) model, we seek to sufficiently utilize LLMs’ generative ability by QL maximization. However, instead of ranking documents with QL estimation, we introduce an auxiliary task of QL maximization to yield a better backbone for contrastively learning a discriminative retriever. We name our model as LLM-QL. To condense global document semantics to a single vector during QL modeling, LLM-QL has two major components, Attention Stop (AS) and Input Corruption (IC). AS stops the attention of predictive tokens to previous tokens until the ending token of the document. IC masks a portion of tokens in the input documents during prediction. Experiments on MSMARCO show that LLM-QL can achieve significantly better performance than other LLM-based retrievers and using QL estimated by LLM-QL for ranking outperforms word-based QL by a large margin. 



# Download dataset 
We utilize the [MSMARCO v1 and TREC-DL](https://microsoft.github.io/msmarco/Datasets) and  [BEIR](https://github.com/beir-cellar/beir).  
We use the hard negative samples provided by [Tevatron](https://www.dropbox.com/scl/fi/pkm1mtgfobae9kuesp7dr/train-tevatron.jsonl?rlkey=2thutc4zkozr9jp4zbbrz5rvi&dl=0). 


# Run
## QL Modeling
```
Deepspeed --num_gpus=8 ql-learning/run.py \
  --deepspeed ql-learning/ds_zero3_config.json \
  --output_dir $encode_path1 \
  --model_name_or_path $ori_model_path1 \
  --train_data $pre_train_path1 \
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
The trained model can be directly downloaded on (QL)[https://huggingface.co/hengranZhang/LLM-QL/tree/main]
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
The trained model can be directly downloaded on (QL)[https://huggingface.co/hengranZhang/LLM-QL/tree/main]

