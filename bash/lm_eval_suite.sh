export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

# UB config
CODE_DIR=/home/csgrad/souvikda/projects/uniwiz/
PRETRAIN_MODEL_DIR=/home/csgrad/souvikda/hf_models/

BACKBONE_MODEL=Mistral-7B-v0.1
LORA_CKPT=processed_kw_no_persona_k_1_n_10_kw_k_5_d_1.pt/checkpoint-2000

#lm_eval --model hf \
#  --model_args pretrained=${PRETRAIN_MODEL_DIR}/${BACKBONE_MODEL},peft=${CODE_DIR}/models/${BACKBONE_MODEL}/${LORA_CKPT} \
#  --tasks truthfulqa --device cuda:0

#accelerate launch --main_process_port 9022 -m lm_eval --model hf --model_args pretrained=/home/csgrad/souvikda/hf_models/Mistral-7B-v0.1,peft=/home/csgrad/souvikda/projects/uniwiz/models/Mistral-7B-v0.1/processed_kw_no_persona_k_1_n_10_kw_k_5_d_1.pt/checkpoint-2000  --tasks hellaswag --num_fewshot 10
#accelerate launch -m lm_eval --model hf --model_args pretrained=/home/csgrad/souvikda/hf_models/Mistral-7B-v0.1,peft=/home/csgrad/souvikda/projects/uniwiz/models/Mistral-7B-v0.1/no_safe_processed_kw_no_persona_n_10_kw_k_5_d_1/checkpoint-5200  --tasks hellaswag --num_fewshot 10
#accelerate launch -m lm_eval --model hf --model_args pretrained=/home/csgrad/souvikda/hf_models/Mistral-7B-v0.1,peft=/home/csgrad/souvikda/projects/uniwiz/models/Mistral-7B-v0.1/no_safe_processed_kw_no_persona_n_10_kw_k_5_d_1/checkpoint-5200  --tasks mmlu --num_fewshot 5
#accelerate launch  -m lm_eval --model hf --model_args pretrained=/home/csgrad/souvikda/hf_models/Mistral-7B-v0.1,peft=/home/csgrad/souvikda/projects/uniwiz/models/Mistral-7B-v0.1/no_safe_processed_kw_no_persona_n_10_kw_k_5_d_1/checkpoint-5200  --tasks winogrande --num_fewshot 5

#accelerate launch --main_process_port 9022 -m lm_eval --model hf --model_args pretrained=/home/csgrad/souvikda/hf_models/Mistral-7B-v0.1,peft=/home/csgrad/souvikda/projects/uniwiz/models/Mistral-7B-v0.1/processed_kw_no_persona_k_1_n_10_kw_k_5_d_1.pt/checkpoint-2000  --tasks arc_challenge  --num_fewshot 25
#accelerate launch --main_process_port 9022 -m lm_eval --model hf --model_args pretrained=/home/csgrad/souvikda/hf_models/Mistral-7B-v0.1,peft=/home/csgrad/souvikda/projects/uniwiz/models/Mistral-7B-v0.1/processed_kw_no_persona_k_1_n_10_kw_k_5_d_1.pt/checkpoint-2000  --tasks hellaswag  --num_fewshot 10
#accelerate launch -m lm_eval --model hf --model_args pretrained=/home/csgrad/souvikda/hf_models/Mistral-7B-v0.1,peft=/home/csgrad/souvikda/projects/uniwiz/models/Mistral-7B-v0.1/processed_kw_no_persona_k_1_n_10_kw_k_5_d_1.pt/checkpoint-2000  --tasks winogrande  --num_fewshot 5

accelerate launch -m lm_eval --model hf --model_args pretrained=/home/csgrad/souvikda/hf_models/Mistral-7B-v0.1,peft=/home/csgrad/souvikda/projects/uniwiz/models/Mistral-7B-v0.1/dpo/checkpoint-400 --tasks gsm8k --num_fewshot 5
