import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from .aesthetics_scorer import preprocess, load_model


def load_aesthetics_and_artifacts_models(device=torch.device("cuda")):
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    vision_model = model.vision_model
    vision_model.to(device)
    del model
    clip_processor = CLIPProcessor.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    )
    rating_model = load_model("aesthetics_scorer_rating_openclip_vit_h_14").to(device)
    artifacts_model = load_model("aesthetics_scorer_artifacts_openclip_vit_h_14").to(
        device
    )
    return vision_model, clip_processor, rating_model, artifacts_model


def compute_aesthetics_and_artifacts_scores(image, models):
    assert isinstance(image, Image.Image)
    assert isinstance(models, tuple) and len(models) == 4
    vision_model, clip_processor, rating_model, artifacts_model = models

    inputs = clip_processor(images=image, return_tensors="pt").to(
        next(vision_model.parameters()).device
    )
    with torch.no_grad():
        vision_output = vision_model(**inputs)
    pooled_output = vision_output.pooler_output
    embedding = preprocess(pooled_output)
    with torch.no_grad():
        rating = rating_model(embedding)
        artifact = artifacts_model(embedding)
    return rating.detach().cpu().item(), artifact.detach().cpu().item()
