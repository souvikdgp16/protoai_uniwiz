export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

# UB config
CODE_DIR=/home/csgrad/souvikda/projects/uniwiz/
DATA_DIR=/home/csgrad/souvikda/projects/uniwiz/dataset/data_annotation/data
DS_CONFIG=/home/csgrad/souvikda/projects/uniwiz/src/ds_config.json
PRETRAIN_MODEL_DIR=/home/csgrad/souvikda/hf_models/



#FILE=processed_kw_no_persona_k_1_n_10_kw_k_5_d_1.pkl
#OP_FILE=processed_kw_no_persona_k_1_n_10_kw_k_5_d_1.pt
#OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/models/llama-7b/
#BACKBONE_MODEL=llama-7b
#
#deepspeed ${CODE_DIR}src/sft/lora_ft.py \
#--dataset_path ${DATA_DIR}/${FILE} \
#--pretrained_model_path ${PRETRAIN_MODEL_DIR}/${BACKBONE_MODEL} \
#--micro_batch_size 8 \
#--batch_size 64 \
#--output_path ${OUTPUT_DIR}/${OP_FILE} \
#--deepspeed ${DS_CONFIG}


#FILE=processed_kw_no_persona_k_1_n_10_kw_k_5_d_1.pkl
#OP_FILE=processed_kw_no_persona_k_1_n_10_kw_k_5_d_1.pt
#OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/models/Mistral-7B-v0.1/
#BACKBONE_MODEL=Mistral-7B-v0.1
#
#deepspeed ${CODE_DIR}src/sft/lora_ft.py \
#--dataset_path ${DATA_DIR}/${FILE} \
#--pretrained_model_path ${PRETRAIN_MODEL_DIR}/${BACKBONE_MODEL} \
#--micro_batch_size 16 \
#--batch_size 64 \
#--learning_rate 5e-7 \
#--output_path ${OUTPUT_DIR}/${OP_FILE} \
#--deepspeed ${DS_CONFIG}




#FILE=processed_kw_no_persona_k_1_n_10_kw_k_5_d_4.pkl
#OP_FILE=processed_kw_no_persona_k_1_n_10_kw_k_5_d_4
#OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/models/Mistral-7B-v0.1/
#BACKBONE_MODEL=Mistral-7B-v0.1
#
#deepspeed ${CODE_DIR}src/sft/lora_ft.py \
#--dataset_path ${DATA_DIR}/${FILE} \
#--pretrained_model_path ${PRETRAIN_MODEL_DIR}/${BACKBONE_MODEL} \
#--micro_batch_size 16 \
#--batch_size 64 \
#--learning_rate 5e-7 \
#--output_path ${OUTPUT_DIR}/${OP_FILE} \
#--deepspeed ${DS_CONFIG}

#FILE=processed_2_kw_no_persona_k_1_n_10_kw_k_5_d_1.pkl
#OP_FILE=no_safe_processed_kw_no_persona_n_10_kw_k_5_d_1_instruct
#OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/models/Mistral-7B-v0.1/
#BACKBONE_MODEL=Mistral-7B-Instruct-v0.2
#
#deepspeed ${CODE_DIR}src/sft/lora_ft.py \
#--dataset_path ${DATA_DIR}/${FILE} \
#--pretrained_model_path ${PRETRAIN_MODEL_DIR}/${BACKBONE_MODEL} \
#--micro_batch_size 16 \
#--batch_size 64 \
#--learning_rate 1e-8 \
#--output_path ${OUTPUT_DIR}/${OP_FILE} \
#--deepspeed ${DS_CONFIG}


FILE=processed_instruct_kw_no_persona_k_1_n_10_kw_k_5_d_1.pkl
OP_FILE=new_instruct_kw_no_persona_k_1_n_10_kw_k_5_d_3
OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/models/Mistral-7B-v0.1/
BACKBONE_MODEL=Mistral-7B-v0.1

deepspeed ${CODE_DIR}src/sft/lora_ft.py \
--dataset_path ${DATA_DIR}/${FILE} \
--pretrained_model_path ${PRETRAIN_MODEL_DIR}/${BACKBONE_MODEL} \
--micro_batch_size 16 \
--batch_size 64 \
--learning_rate 1e-7 \
--output_path ${OUTPUT_DIR}/${OP_FILE} \
--deepspeed ${DS_CONFIG}




