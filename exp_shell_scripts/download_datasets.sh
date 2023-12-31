# Plase run this script in the root directory of this repo

# Datasets
mkdir datasets && cd datasets
mkdir selected && mkdir generated && mkdir watermarked && mkdir source

# Source datasets
cd source
# ImageNet related datasets
mkdir imagenet && cd imagenet
# ImageNet 2012 (ILSVRC2012) val dataset (50K images, 1K classes, 50 images per class)
mkdir imagenet-val && cd imagenet-val
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
cd ..
# Tiny-ImageNet (subset of ImageNet, 200 classes with 500 images each)
wget https://image-net.org/data/tiny-imagenet-200.zip && unzip tiny-imagenet-200.zip && rm tiny-imagenet-200.zip
# ImageNette (subset of 10 classes)
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz && tar -xvzf imagenette2-320.tgz && rm imagenette2-320.tgz
# ImageNet class index (default of guided diffusion)
wget https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json
cd ..
# MS-COCO captioning 2017 dataset (118K+5K images, 5 captions per image)
mkdir mscoco && cd mscoco
wget http://images.cocodataset.org/zips/train2017.zip && unzip train2017.zip && rm train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip && rm val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip
cd ..
# DiffusionDB (2M images, 1.5M prompts, only using the 50K split)
mkdir diffusiondb && cd diffusiondb
python -c 'from datasets import load_dataset; dataset = load_dataset("poloclub/diffusiondb", "2m_random_50k"); dataset.save_to_disk("./2m_random_50k/")'
cd ..
# LAION DALLE3 (6.8K images)
mkdir dalle3 && cd dalle3
# There is a bug in downloading this dataset now, see https://huggingface.co/datasets/laion/dalle-3-dataset/discussions/7
# python -c 'from datasets import load_dataset; dataset = load_dataset("laion/dalle-3-dataset"); dataset.save_to_disk(".")'
cd ..

# Generated datasets
# Guided diffusion generated dataset
cd generated
wget --output-document=imagenet_guided_64_tiny-imagenet_50.zip "https://www.dropbox.com/scl/fi/aixqtm6uwpnxlr6gih6zv/imagenet_guided_64_tiny-imagenet_50.zip?rlkey=227jqa72d3kotcwpkre5857pa&dl=1" && unzip imagenet_guided_64_tiny-imagenet_50.zip && rm imagenet_guided_64_tiny-imagenet_50.zip
wget --output-document=imagenet_guided_256_imagenette_100.zip "https://www.dropbox.com/scl/fi/zri55f21dssf2q406jgdx/imagenet_guided_256_imagenette_100.zip?rlkey=ynxe5omb1qpx97s44evmcnsrf&dl=1" && unzip imagenet_guided_256_imagenette_100.zip && rm imagenet_guided_256_imagenette_100.zip
cd ..
