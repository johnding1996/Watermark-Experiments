# Plase run this script in the root directory of this repo

# Dependencies
# Install and upgrade jupyter
pip install --upgrade pip ipython jupyter ipywidgets
# Install dependences (on CUDA 12.1)
pip install torch torchvision torchaudio
pip install transformers diffusers accelerate ftfy
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install torchattacks datasets pycocotools scikit-learn scikit-image matplotlib imageio

# Logins
# Setup git (remove later)
git config --global user.name "mcding" && git config --global user.email "mcding@umd.edu"
# Setup wandb (remove later)
pip install wandb
wandb login e66105607bb979a5e6e49a3d5d4ce02894398354

# Datasets
mkdir datasets && cd datasets
# Real-world image datasets
# ImageNet related datasets
mkdir imagenet-related && cd imagenet
# ImageNet 2012 (ILSVRC2012) val dataset (50,000 images, 1,000 classes, 50 images per class)
mkdir imagenet-val && cd imagenet-val
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
cd ..
# Tiny-ImageNet (subset of ImageNet, 200 classes with 500 images each)
wget https://image-net.org/data/tiny-imagenet-200.zip && unzip tiny-imagenet-200.zip && rm tiny-imagenet-200.zip
# ImageNette (subset of 10 classes from ImageNet)
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz && tar -xvzf imagenette2-320.tgz && rm imagenette2-320.tgz
# ImageNet class index (default of guided diffusion)
wget https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json
cd ..
# MS-COCO captioning 2017 val dataset (5,000 images, 5 captions per image)
mkdir ms-coco && cd ms-coco
wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip && rm val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip
cd ..
# Diffusion generated images
# Guided diffusion generated images
mkdir guided-diffusion && cd guided-diffusion
# Guided diffusion checkpoints
wget --output-document=imagenet_guided_64_tiny-imagenet_50.zip "https://www.dropbox.com/scl/fi/aixqtm6uwpnxlr6gih6zv/imagenet_guided_64_tiny-imagenet_50.zip?rlkey=227jqa72d3kotcwpkre5857pa&dl=1" && unzip imagenet_guided_64_tiny-imagenet_50.zip && rm imagenet_guided_64_tiny-imagenet_50.zip
cd ..


wget --output-document=tree_ring_guided_1k_1m.zip "https://www.dropbox.com/scl/fi/hbw4d853t0hr0tz9o3tjr/tree_ring_guided_1k_1m.zip?rlkey=jmc2hegj1k2deqkm6h7uq9awu&dl=1" && unzip tree_ring_guided_1k_1m.zip && rm tree_ring_guided_1k_1m.zip
cd ..

# Models
mkdir models && cd models
# Guided diffusion checkpoints
mkdir guided_diffusion && cd guided_diffusion
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_classifier.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_diffusion.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_classifier.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64_256_upsampler.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128_512_upsampler.pt
cd ..
# StegaStamp checkpoints
mkdir stegastamp && cd stegastamp
wget http://people.eecs.berkeley.edu/~tancik/stegastamp/saved_models.tar.xz && tar -xJf saved_models.tar.xz && rm saved_models.tar.xz && mv saved_models/stegastamp_pretrained/ tensorflow && rm -rf saved_models
cd ..
