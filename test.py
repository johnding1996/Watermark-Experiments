from datasets import load_from_disk
from metrics import *

dataset = load_from_disk("/scratch0/mcding/mscoco_5k")
images1 = [dataset[i]["image"] for i in range(2500)]
images2 = [dataset[i]["image"] for i in range(2500, 5000)]

fid_mean, fid_std = compute_fid_repeated(images1, images2, verbose=True)
print(fid_mean, fid_std)
