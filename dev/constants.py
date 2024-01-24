from .io import decode_array_from_string

LIMIT, SUBSET_LIMIT = 5000, 1000

DATASET_NAMES = {
    "diffusiondb": "DiffusionDB",
    "mscoco": "MS-COCO",
    "dalle3": "DALL-E 3",
}

WATERMARK_METHODS = {
    "tree_ring": "Tree-Ring",
    "stable_sig": "Stable-Signature",
    "stegastamp": "Stega-Stamp",
}

PERFORMANCE_METRICS = {
    "acc_1": "Mean Detection Accuracy",
    "auc_1": "AUC",
    "low100_1": "TPR@1%FPR",
    "low1000_1": "TPR@0.1%FPR",
    "acc_100": "Identification Accuracy (100 Users)",
    "acc_1000": "Identification Accuracy (1K Users)",
    "acc_1000000": "Identification Accuracy (1M Users)",
}

QUALITY_METRICS = {
    "legacy_fid": "FID",
    "clip_fid": "CLIP FID",
    "psnr": "PSNR",
    "ssim": "SSIM",
    "nmi": "Normed Mutual-Info",
    "lpips": "LPIPS",
    "watson": "Watson-DFT",
    "aesthetics": "Delta Aesthetics",
    "artifacts": "Delta Artifacts",
    "clip_score": "Delta CLIP-Score",
}

EVALUATION_SETUPS = {
    "removal": "Removal",
    "spoofing": "Spoofing",
    "combined": "Combined",
}

GROUND_TRUTH_MESSAGES = {
    "tree_ring": decode_array_from_string(
        "H4sIALRwUmUC/42SvYrCQBSFLW18iam3EZcUFgErkYBFSLcYENZgISgoiMjCVj6FzyFCCoUlTQgrE8TnkcNhcEZIbjhw7x3Ox/zdu1fr+XQ1U/0vr/c5+VDfmx1WKlksp5uup35anX+jLHrVWFHX0FS2cw2h8x+z69KBYs1MxvVjDXkPZpvh/iC8B1SUzKTMWTZTlEV5iBBtzqVAHKJEI4KrphL9OwInU1AzUqbku9W/s+mf1f+93D+5/1Vz8z5j+djIv79qrKj2wFS20x5Al5LZdelApxszGdc/3aBUM9sM9weRasgPmEmZs2zGD/xgmyPanEuB2ObHDBFcNXXMwiE4mYKakTIl363+nU3/rP7v5f7J/a+am/cZewKA1ipNFgUAAA=="
    ),
    "stable_sig": decode_array_from_string(
        "H4sIADtrUmUC/6tWKs5ILEhVsoo2sYjVUUopqQRxlJLy83OUahkYGRkZgBBMMIAAhAvmwUUZIRIgcQBxGJ0kTgAAAA=="
    ),
    "stegastamp": decode_array_from_string(
        "H4sIAGRrUmUC/6tWKs5ILEhVsoo2NDCI1VFKKakE8ZSS8vNzlGoZGBkZGRhAmAFCgxGIC+dBCAaYEEyCEVkOzISrYIToZIAoY2AEAG5jy4ODAAAA"
    ),
}


ATTACK_NAMES = {
    "distortion_single_rotation": "Dist-Rotation",
    "distortion_single_resizedcrop": "Dist-RCrop",
    "distortion_single_erasing": "Dist-Erase",
    "distortion_single_brightness": "Dist-Bright",
    "distortion_single_contrast": "Dist-Contrast",
    "distortion_single_blurring": "Dist-Blur",
    "distortion_single_noise": "Dist-Noise",
    "distortion_single_jpeg": "Dist-JPEG",
    "distortion_combo_geometric": "DistCom-Geo",
    "distortion_combo_photometric": "DistCom-Photo",
    "distortion_combo_degradation": "DistCom-Deg",
    "distortion_combo_all": "DistCom-All",
    "regen_diffusion": "Regen-Diff",
    "regen_diffusion_prompt": "Regen-DiffP",
    "regen_vae": "Regen-VAE",
    "kl_vae": "Regen-KLVAE",
    "2x_regen": "Rinse-2xDiff",
    "4x_regen": "Rinse-4xDiff",
    "4x_regen_bmshj": "RinseD-VAE",
    "4x_regen_kl_vae": "RinseD-KLVAE",
    "adv_emb_same_vae_untg": "AdvEmbG-KLVAE8",
    "adv_emb_resnet18_untg": "AdvEmbB-RN18",
    "adv_emb_clip_untg_alphaRatio_0.05_step_200": "AdvEmbB-CLIP",
    "adv_emb_klf16_vae_untg": "AdvEmbB-KLVAE16",
    "adv_emb_sdxl_vae_untg": "AdvEmbB-SdxlVAE",
    "adv_cls_unwm_wm_0.01_50_warm_train3k": "AdvCls-UnWM&WM",
    "adv_cls_real_wm_0.01_50_warm": "AdvCls-Real&WM",
    "adv_cls_wm1_wm2_0.01_50_warm": "AdvCls-WM1&WM2",
}


ATTACK_CATEGORIES = {
    "Distortion Single": [
        "distortion_single_rotation",
        "distortion_single_resizedcrop",
        "distortion_single_erasing",
        "distortion_single_brightness",
        "distortion_single_contrast",
        "distortion_single_blurring",
        "distortion_single_noise",
        "distortion_single_jpeg",
    ],
    "Distortion Combination": [
        "distortion_combo_geometric",
        "distortion_combo_photometric",
        "distortion_combo_degradation",
        "distortion_combo_all",
    ],
    "Regeneration Single": ["regen_diffusion", "kl_vae"],
    "Regeneration Rinsing": ["2x_regen", "4x_regen"],
    "Adv Embedding Gray-box": ["adv_emb_same_vae_untg"],
    "Adv Embedding Black-box": [
        "adv_emb_clip_untg_alphaRatio_0.05_step_200",
        "adv_emb_sdxl_vae_untg",
        "adv_emb_klf16_vae_untg",
    ],
    "Adv Surrogate Detector": ["adv_cls_wm1_wm2_0.01_50_warm"],
}


QUALITY_NORMALIZATION_THRESHOLDS = (0.1, 0.9)


QUALITY_NORMALIZATION_INTERVALS = {
    "legacy_fid": (1.2018279835082921, 53.4025023653088),
    "clip_fid": (0.28221379251059064, 19.01774805091691),
    "psnr": (41.97326692759963, 12.146514692824278),
    "ssim": (0.9794436656191583, 0.32230828785622945),
    "nmi": (1.706872784927291, 1.0645803311654816),
    "lpips": (0.01795978478139732, 0.6205694336295128),
    "aesthetics": (0.024198556509613993, 1.8233248041778802),
    "artifacts": (0.0018974746170453707, -0.581003303605318),
}


QUALITY_NORMALIZATION_WEIGHTS = {
    "legacy_fid": 1 / 4 / 2,
    "clip_fid": 1 / 4 / 2,
    "psnr": 1 / 4 / 3,
    "ssim": 1 / 4 / 3,
    "nmi": 1 / 4 / 3,
    "lpips": 1 / 4 / 1,
    "aesthetics": 1 / 4 / 2,
    "artifacts": 1 / 4 / 2,
}
