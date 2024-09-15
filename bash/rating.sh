# UB config
CODE_DIR=/home/csgrad/souvikda/projects/uniwiz/
OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/dataset/data_annotation/data
DATA_DIR=/home/csgrad/souvikda/projects/uniwiz/dataset/data_annotation/data

IN_FILE=new_kw_no_persona_k_1_n_10_kw_k_5_d_3_v2.pkl
OP_FILE=new_kw_no_persona_k_1_n_10_kw_k_5_d_3_v2_rating_c2.pkl

python3 ${CODE_DIR}dataset/data_annotation/rating.py \
--raw_dataset_path ${DATA_DIR}/${IN_FILE} \
--model claude-2 \
--output_path ${OUTPUT_DIR}/${OP_FILE}

OP_FILE=new_kw_no_persona_k_1_n_10_kw_k_5_d_3_v2_rating_ci1.2.pkl

python3 ${CODE_DIR}dataset/data_annotation/rating.py \
--raw_dataset_path ${DATA_DIR}/${IN_FILE} \
--model claude-instant-1.2 \
--output_path ${OUTPUT_DIR}/${OP_FILE}
