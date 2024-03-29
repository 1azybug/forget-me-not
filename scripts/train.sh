# ensure in forget-me-not dir
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/1/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluate.py --config_path ./configs/llama/1/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/shuffle0/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluate.py --config_path ./configs/llama/shuffle0/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/order0/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluate.py --config_path ./configs/llama/order0/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/seglen2048/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluate.py --config_path ./configs/llama/seglen2048/config.json

# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/250ktokens_order/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluate.py --config_path ./configs/llama/250ktokens_order/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/250ktokens_shuffle/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluate.py --config_path ./configs/llama/250ktokens_shuffle/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/seglen2048_order/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluate.py --config_path ./configs/llama/seglen2048_order/config.json





# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/128ktokens_shuffle/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluator.py --config_path ./configs/llama/128ktokens_shuffle/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/64ktokens_shuffle/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluator.py --config_path ./configs/llama/64ktokens_shuffle/config.json

# 记得改config里的save_dir


# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/lr_1e-3/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluator.py --config_path ./configs/llama/lr_1e-3/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/lr_1e-2/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluator.py --config_path ./configs/llama/lr_1e-2/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/lr_1e-1/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluator.py --config_path ./configs/llama/lr_1e-1/config.json

# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/w500s0.7/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluator.py --config_path ./configs/llama/w500s0.7/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/w500s0.9/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluator.py --config_path ./configs/llama/w500s0.9/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/w500s0.5/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluator.py --config_path ./configs/llama/w500s0.5/config.json


# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/patience100/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluator.py --config_path ./configs/llama/patience100/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/patience500/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluator.py --config_path ./configs/llama/patience500/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/patience1000/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluator.py --config_path ./configs/llama/patience1000/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/patience2000/config.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluator.py --config_path ./configs/llama/patience2000/config.json

CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/llama/cosine/config.json
CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluator.py --config_path ./configs/llama/cosine/config.json

CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train.py --config_path ./configs/modify/keynorm/config.json
CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluator.py --config_path ./configs/modify/keynorm/config.json