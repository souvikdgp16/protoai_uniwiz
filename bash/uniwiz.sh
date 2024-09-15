# UB config
CODE_DIR=/home/csgrad/souvikda/projects/uniwiz/
OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/dataset/data_annotation/data
DATA_DIR=/home/csgrad/souvikda/projects/uniwiz/dataset/data_annotation/data
#PRETRAIN_MODEL_DIR=/home/csgrad/souvikda/hf_models/
#BACKBONE_MODEL=llama-7b
FACT=wow_knowledge.pkl
PRIMING=priming_v2.pkl
NAMES=names.txt


#OP_FILE=kw_no_persona_k_1_n_10_kw_k_5_d_1.pkl
#
#python3 ${CODE_DIR}dataset/data_annotation/knowledge_injection.py \
#--fact_path ${DATA_DIR}/${FACT} \
#--safety_primed_conv_path ${DATA_DIR}/${PRIMING} \
#--names_path ${DATA_DIR}/${NAMES} \
#--k 1 \
#--n 10 \
#--kw_k 5 \
#--depth 1 \
#--output_path ${OUTPUT_DIR}/${OP_FILE}
#
#
#OP_FILE=kw_no_persona_k_2_n_10_kw_k_5_d_1.pkl
#
#python3 ${CODE_DIR}dataset/data_annotation/knowledge_injection.py \
#--fact_path ${DATA_DIR}/${FACT} \
#--safety_primed_conv_path ${DATA_DIR}/${PRIMING} \
#--names_path ${DATA_DIR}/${NAMES} \
#--k 2 \
#--n 10 \
#--kw_k 5 \
#--depth 1 \
#--output_path ${OUTPUT_DIR}/${OP_FILE}
#
#OP_FILE=kw_no_persona_k_3_n_10_kw_k_5_d_1.pkl
#
#python3 ${CODE_DIR}dataset/data_annotation/knowledge_injection.py \
#--fact_path ${DATA_DIR}/${FACT} \
#--safety_primed_conv_path ${DATA_DIR}/${PRIMING} \
#--names_path ${DATA_DIR}/${NAMES} \
#--k 3 \
#--n 10 \
#--kw_k 5 \
#--depth 1 \
#--output_path ${OUTPUT_DIR}/${OP_FILE}
#
#OP_FILE=kw_no_persona_k_4_n_10_kw_k_5_d_1.pkl
#
#python3 ${CODE_DIR}dataset/data_annotation/knowledge_injection.py \
#--fact_path ${DATA_DIR}/${FACT} \
#--safety_primed_conv_path ${DATA_DIR}/${PRIMING} \
#--names_path ${DATA_DIR}/${NAMES} \
#--k 4 \
#--n 10 \
#--kw_k 5 \
#--depth 1 \
#--output_path ${OUTPUT_DIR}/${OP_FILE}
#
#OP_FILE=kw_no_persona_k_5_n_10_kw_k_5_d_1.pkl
#
#python3 ${CODE_DIR}dataset/data_annotation/knowledge_injection.py \
#--fact_path ${DATA_DIR}/${FACT} \
#--safety_primed_conv_path ${DATA_DIR}/${PRIMING} \
#--names_path ${DATA_DIR}/${NAMES} \
#--k 5 \
#--n 10 \
#--kw_k 5 \
#--depth 1 \
#--output_path ${OUTPUT_DIR}/${OP_FILE}



#OP_FILE=kw_no_persona_k_1_n_10_kw_k_5_d_2.pkl
#
#python3 ${CODE_DIR}dataset/data_annotation/knowledge_injection.py \
#--fact_path ${DATA_DIR}/${FACT} \
#--safety_primed_conv_path ${DATA_DIR}/${PRIMING} \
#--names_path ${DATA_DIR}/${NAMES} \
#--k 1 \
#--n 10 \
#--kw_k 5 \
#--depth 2 \
#--output_path ${OUTPUT_DIR}/${OP_FILE}

#OP_FILE=kw_no_persona_k_1_n_10_kw_k_5_d_3.pkl
#
#python3 ${CODE_DIR}dataset/data_annotation/knowledge_injection.py \
#--fact_path ${DATA_DIR}/${FACT} \
#--safety_primed_conv_path ${DATA_DIR}/${PRIMING} \
#--names_path ${DATA_DIR}/${NAMES} \
#--k 1 \
#--n 10 \
#--kw_k 5 \
#--depth 3 \
#--output_path ${OUTPUT_DIR}/${OP_FILE}
#
#OP_FILE=kw_no_persona_k_1_n_10_kw_k_5_d_4.pkl
#
#python3 ${CODE_DIR}dataset/data_annotation/knowledge_injection.py \
#--fact_path ${DATA_DIR}/${FACT} \
#--safety_primed_conv_path ${DATA_DIR}/${PRIMING} \
#--names_path ${DATA_DIR}/${NAMES} \
#--k 1 \
#--n 10 \
#--kw_k 5 \
#--depth 4 \
#--output_path ${OUTPUT_DIR}/${OP_FILE}
#
#OP_FILE=kw_no_persona_k_5_n_10_kw_k_5_d_5.pkl
#
#python3 ${CODE_DIR}dataset/data_annotation/knowledge_injection.py \
#--fact_path ${DATA_DIR}/${FACT} \
#--safety_primed_conv_path ${DATA_DIR}/${PRIMING} \
#--names_path ${DATA_DIR}/${NAMES} \
#--k 1 \
#--n 10 \
#--kw_k 5 \
#--depth 5 \
#--output_path ${OUTPUT_DIR}/${OP_FILE}


#OP_FILE=kw_no_persona_k_1_n_10_kw_k_5_d_1.pkl
#
#python3 ${CODE_DIR}dataset/data_annotation/knowledge_injection.py \
#--fact_path ${DATA_DIR}/${FACT} \
#--safety_primed_conv_path ${DATA_DIR}/${PRIMING} \
#--names_path ${DATA_DIR}/${NAMES} \
#--k 1 \
#--n 10 \
#--kw_k 5 \
#--depth 2 \
#--output_path ${OUTPUT_DIR}/${OP_FILE}

OP_FILE=new_kw_no_persona_k_1_n_10_kw_k_5_d_3_v2.pkl

python3 ${CODE_DIR}dataset/data_annotation/knowledge_injection.py \
--fact_path ${DATA_DIR}/${FACT} \
--safety_primed_conv_path ${DATA_DIR}/${PRIMING} \
--names_path ${DATA_DIR}/${NAMES} \
--k 1 \
--n 10 \
--kw_k 5 \
--depth 5 \
--output_path ${OUTPUT_DIR}/${OP_FILE}
