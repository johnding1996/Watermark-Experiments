# Upgrade jupyter
pip install --upgrade pip ipython jupyter ipywidgets
# Install dependences (CUDA 12.1)
pip install torch torchvision torchaudio
pip install transformers diffusers datasets scikit-learn matplotlib imageio
# Download datasets
mkdir datasets && cd datasets
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz && tar -xvzf imagenette2-320. && rm imagenette2-320.tgz
wget https://image-net.org/data/tiny-imagenet-200.zip && unzip tiny-imagenet-200.zip && rm tiny-imagenet-200.zip
cd ..
# Download checkpoints
mkdir models && cd models
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt