# Plase run this script in the root directory of this repo

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
