export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

# UB config
CODE_DIR=/home/csgrad/souvikda/projects/uniwiz/
PRETRAIN_MODEL_DIR=/home/csgrad/souvikda/hf_models/

#DATA_DIR=/home/csgrad/souvikda/projects/uniwiz/models/llama-7b/
#BACKBONE_MODEL=llama-7b
#LORA_FILE=processed_kw_no_persona_k_1_n_10_kw_k_5_d_1.pt/checkpoint-5357/
#OP_FILE=tfqa_processed_kw_no_persona_k_1_n_10_kw_k_5_d_1.txt
#OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/models/llama-7b/
#
#python3 ${CODE_DIR}src/inference/tfqa.py \
#--pretrained_model_path ${PRETRAIN_MODEL_DIR}/${BACKBONE_MODEL} \
#--lora_weights_path ${DATA_DIR}${LORA_FILE} \
#--output_path ${OUTPUT_DIR}/${OP_FILE}

#BACKBONE_MODEL=Mistral-7B-v0.1
#LORA_FILE=processed_kw_no_persona_k_1_n_10_kw_k_5_d_1.pt/checkpoint-2000/
#OP_FILE=tfqa_processed_kw_no_persona_k_1_n_10_kw_k_5_d_1.txt
#OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/models/Mistral-7B-v0.1/
#DATA_DIR=${CODE_DIR}/models/${BACKBONE_MODEL}/
#
#python3 ${CODE_DIR}src/inference/tfqa.py \
#--pretrained_model_path ${PRETRAIN_MODEL_DIR}/${BACKBONE_MODEL} \
#--lora_weights_path ${DATA_DIR}${LORA_FILE} \
#--output_path ${OUTPUT_DIR}/${OP_FILE}

#BACKBONE_MODEL=Mistral-7B-v0.1
#LORA_FILE=processed_kw_no_persona_k_1_n_10_kw_k_5_d_4/checkpoint-2000/
#OP_FILE=tfqa_processed_kw_no_persona_k_1_n_10_kw_k_5_d_4.txt
#OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/models/Mistral-7B-v0.1/
#DATA_DIR=${CODE_DIR}/models/${BACKBONE_MODEL}/
#
#python3 ${CODE_DIR}src/inference/tfqa.py \
#--pretrained_model_path ${PRETRAIN_MODEL_DIR}/${BACKBONE_MODEL} \
#--lora_weights_path ${DATA_DIR}${LORA_FILE} \
#--output_path ${OUTPUT_DIR}/${OP_FILE}


BACKBONE_MODEL=Mistral-7B-v0.1
LORA_FILE=processed_kw_no_persona_n_10_kw_k_5_d_1/checkpoint-5000/
OP_FILE=tfqa_processed_kw_no_persona_n_10_kw_k_5_d_1.txt
OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/models/Mistral-7B-v0.1/
DATA_DIR=${CODE_DIR}/models/${BACKBONE_MODEL}/

python3 ${CODE_DIR}src/inference/tfqa.py \
--pretrained_model_path ${PRETRAIN_MODEL_DIR}/${BACKBONE_MODEL} \
--lora_weights_path ${DATA_DIR}${LORA_FILE} \
--output_path ${OUTPUT_DIR}/${OP_FILE}