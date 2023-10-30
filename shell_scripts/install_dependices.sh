# Plase run this script in the root directory of this repo

# Logins (remove later)
# Setup git
git config --global user.name "mcding" && git config --global user.email "mcding@umd.edu"
# Setup wandb
pip install wandb && wandb login e66105607bb979a5e6e49a3d5d4ce02894398354

# Dependencies
# Install and upgrade jupyter
pip install --upgrade pip ipython jupyter ipywidgets
# Install dependences (on CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers 'datasets[vision]' open_clip_torch ftfy
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
pip install torchattacks pycocotools scikit-learn scikit-image matplotlib imageio
pip install accelerate deepspeed
