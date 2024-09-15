export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# UB config
CODE_DIR=/home/csgrad/souvikda/projects/uniwiz/
PRETRAIN_MODEL_DIR=/home/csgrad/souvikda/hf_models/
DS_CONFIG=/home/csgrad/souvikda/projects/uniwiz/src/ds_config.json

BACKBONE_MODEL=Mistral-7B-v0.1
OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/models/
#LORA_CKPT=processed_kw_no_persona_k_1_n_10_kw_k_5_d_1.pt/checkpoint-2000
LORA_CKPT=new_instruct_kw_no_persona_k_1_n_10_kw_k_5_d_3/checkpoint-5400
OP_FOLDER=dpo

python3 ${CODE_DIR}src/dpo/dpo_runner.py \
--model_name_or_path ${PRETRAIN_MODEL_DIR}/${BACKBONE_MODEL} \
--lora_weight_path ${OUTPUT_DIR}/${BACKBONE_MODEL}/${LORA_CKPT} \
--learning_rate 3e-5 \
--output_path ${OUTPUT_DIR}/${BACKBONE_MODEL}/${OP_FOLDER} \
