# forget-me-not
Try to give AI a longer life


# 环境配置
```
git clone https://github.com/1azybug/forget-me-not.git
cd forget-me-not
conda create -n forget python=3.10 -y
conda activate forget
# [cuda12.1 用于 xformers安装] https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers
conda install pytorch==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
# (Optional) Testing the installation
python -m xformers.info

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
