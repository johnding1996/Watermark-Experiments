# Plase run this script in the root directory of this repo

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Dependencies
# Install and upgrade jupyter``
pip install --upgrade pip ipython jupyter ipywidgets python-dotenv
# Install dependences (on CUDA 11.8)
# PyTorch 2.1.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Huggingface libraries
pip install transformers diffusers 'datasets[vision]' open_clip_torch ftfy
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
# Other machine learning libraries
pip install torchattacks scikit-learn scikit-image
# Data processing libraries
pip install pycocotools matplotlib imageio opencv-python
# Metric libraries
pip install git+https://github.com/openai/CLIP.git
# Parallel libraries
pip install accelerate deepspeed
# Other libraries


# Logins (remove later)
# Setup git
git config --global user.name "mcding" && git config --global user.email "mcding@umd.edu"
# Setup wandb
pip install wandb && wandb login e66105607bb979a5e6e49a3d5d4ce02894398354

# Fix CUDNN issue for libnvrtc.so, see https://stackoverflow.com/questions/76216778/userwarning-applied-workaround-for-cudnn-issue-install-nvrtc-so
cd venv/lib/python3.10/site-packages/torch/lib
ln -s libnvrtc-*.so.11.2 libnvrtc.so
cd -

# Fix vscode jupyter issue, see https://github.com/microsoft/vscode-jupyter/issues/14618
pip install ipython==8.16.1