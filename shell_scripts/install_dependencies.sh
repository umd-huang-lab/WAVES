# Plase run this script in the root directory of this repo

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Dependencies
# Install and upgrade jupyter``
pip install --upgrade pip ipython jupyter ipywidgets python-dotenv
# Install dependences (on CUDA 11.8)
# PyTorch 2.1.0
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Huggingface libraries
pip install transformers diffusers 'datasets[vision]' ftfy
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
# Other machine learning libraries
pip install onnx onnxruntime-gpu torchmetrics open_clip_torch torchattacks scikit-learn scikit-image pandas
# Data processing libraries
pip install pycocotools matplotlib imageio opencv-python
# Metric libraries
pip install git+https://github.com/openai/CLIP.git
# Parallel libraries
pip install accelerate deepspeed
# HF space and gradio libraries
pip install huggingface-hub gitpython gradio==4.3.0 plotly plotly-express wordcloud
# Other libraries

# Fix CUDNN issue for libnvrtc.so, see https://stackoverflow.com/questions/76216778/userwarning-applied-workaround-for-cudnn-issue-install-nvrtc-so
cd venv/lib/python3.10/site-packages/torch/lib
ln -s libnvrtc-*.so.11.2 libnvrtc.so
cd -

# Fix vscode jupyter issue, see https://github.com/microsoft/vscode-jupyter/issues/14618
pip install ipython==8.16.1