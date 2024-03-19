# ensure in forget-me-not dir
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/1/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluate.py --config_path ./configs/llama/1/config.json
CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/shuffle0/config.json
CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluate.py --config_path ./configs/llama/shuffle0/config.json