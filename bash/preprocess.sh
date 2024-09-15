# UB config
CODE_DIR=/home/csgrad/souvikda/projects/uniwiz/
OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/dataset/data_annotation/data
DATA_DIR=/home/csgrad/souvikda/projects/uniwiz/dataset/data_annotation/data
#PRETRAIN_MODEL_DIR=/home/csgrad/souvikda/hf_models/
#BACKBONE_MODEL=llama-7b

#RAW_FILE=kw_no_persona_k_1_n_10_kw_k_5_d_1.pkl
#OP_FILE=processed_kw_no_persona_k_1_n_10_kw_k_5_d_1.pkl
#
#python3 ${CODE_DIR}src/sft/preprocess.py \
#--raw_dataset_path ${DATA_DIR}/${RAW_FILE} \
#--output_path ${DATA_DIR}/${OP_FILE}

#RAW_FILE_1=kw_no_persona_k_1_n_10_kw_k_5_d_1.pkl
#RAW_FILE_2=kw_no_persona_k_2_n_10_kw_k_5_d_1.pkl
#RAW_FILE_3=kw_no_persona_k_3_n_10_kw_k_5_d_1.pkl

RAW_FILE_1=new_kw_no_persona_k_1_n_10_kw_k_5_d_3.pkl
RAW_FILE_2=instruct_kw_no_persona_k_1_n_10_kw_k_5_d_3.pkl

#OP_FILE=processed_kw_no_persona_n_10_kw_k_5_d_1.pkl
OP_FILE=processed_instruct_kw_no_persona_k_1_n_10_kw_k_5_d_1.pkl

#python3 ${CODE_DIR}src/sft/preprocess.py \
#  --raw_dataset_paths ${DATA_DIR}/${RAW_FILE_1} ${DATA_DIR}/${RAW_FILE_2} ${DATA_DIR}/${RAW_FILE_3} \
#  --output_path ${DATA_DIR}/${OP_FILE}


python3 ${CODE_DIR}src/sft/preprocess.py \
  --raw_dataset_paths ${DATA_DIR}/${RAW_FILE_1} \
  --output_path ${DATA_DIR}/${OP_FILE}
