export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

# UB config
CODE_DIR=/home/csgrad/souvikda/projects/uniwiz/


BACKBONE_MODEL=albert-xxlarge-v2
OUTPUT_DIR=/home/csgrad/souvikda/projects/uniwiz/models/
OP_FOLDER=arg_quality

python3 ${CODE_DIR}src/qc/argument_clf.py \
--output_path ${OUTPUT_DIR}/${BACKBONE_MODEL}/${OP_FOLDER} \
