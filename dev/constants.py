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
    "acc_1": "Mean Accuracy",
    "auc_1": "AUC",
    "low100_1": "TPR@1%FPR",
    "low1000_1": "TPR@0.1%FPR",
}

QUALITY_METRICS = {
    "legacy_fid": "Legacy FID",
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
    "combined": "Combined",
    "removal": "Removal",
    "spoofing": "Spoofing",
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
    "distortion_combo_geometric": "Dist-Com-Geo",
    "distortion_combo_photometric": "Dist-Com-Photo",
    "distortion_combo_degradation": "Dist-Com-Deg",
    "distortion_combo_all": "Dist-Com-All",
    "regen_diffusion": "Regen-Diffusion",
    "regen_diffusion_prompt": "Regen-Diffusion&P",
    "regen_vae": "Regen-VAE",
    "kl_vae": "Regen-KLVAE",
    "2x_regen": "Regen-2xDiffusion",
    "4x_regen": "Regen-4xDiffusion",
    "4x_regen_bmshj": "Regen-4xVAE",
    "4x_regen_kl_vae": "Regen-4xKLVAE",
    "adv_emb_resnet18_untg": "AdvEmb-RN18",
    "adv_emb_clip_untg_alphaRatio_0.05_step_200": "AdvEmb-CLIP",
    "adv_emb_same_vae_untg": "AdvEmb-KLVAE8",
    "adv_emb_klf16_vae_untg": "AdvEmb-KLVAE16",
    "adv_emb_sdxl_vae_untg": "AdvEmb-SdxlVAE",
    "adv_cls_unwm_wm_0.01_50_warm_train3k": "AdvCls-UnWM-WM",
    "adv_cls_real_wm_0.01_50_warm": "AdvCls-Real-WM",
    "adv_cls_wm1_wm2_0.01_50_warm": "AdvCls-WM1-WM2",
    "adv_cls_wm1_wm2_0.04_200_warm": "abandon",
}
