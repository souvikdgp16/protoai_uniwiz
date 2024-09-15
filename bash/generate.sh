export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

# UB config
CODE_DIR=/home/csgrad/souvikda/projects/uniwiz/
PRETRAIN_MODEL_DIR=/home/csgrad/souvikda/hf_models/

BACKBONE_MODEL=uniwiz-7B-v0.2
OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/data/
OP_FILE=${BACKBONE_MODEL}.pkl

python3 ${CODE_DIR}dataset/data_annotation/priming.py \
--model ${PRETRAIN_MODEL_DIR}/${BACKBONE_MODEL} \
--output_path ${OUTPUT_DIR}/${OP_FILE} \

BACKBONE_MODEL=Llama-2-7b-chat-hf
OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/data/
OP_FILE=${BACKBONE_MODEL}.pkl

python3 ${CODE_DIR}dataset/data_annotation/priming.py \
--model ${PRETRAIN_MODEL_DIR}/${BACKBONE_MODEL} \
--output_path ${OUTPUT_DIR}/${OP_FILE} \


BACKBONE_MODEL=llama-7b
OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/data/
OP_FILE=${BACKBONE_MODEL}.pkl

python3 ${CODE_DIR}dataset/data_annotation/priming.py \
--model ${PRETRAIN_MODEL_DIR}/${BACKBONE_MODEL} \
--output_path ${OUTPUT_DIR}/${OP_FILE} \
