from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
import torch

device = "cuda:3"
print('prepare models...')
paint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=False
).to(device)
print('finish preparing')

# print('prepare models...')
# image2image_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4",
#     revision="fp16", 
#     torch_dtype=torch.float16,
#     use_auth_token=False
# ).to(device)
# print('finish preparing')

image2image_pipeline = StableDiffusionImg2ImgPipeline(
    vae = paint_pipeline.vae,
    text_encoder = paint_pipeline.text_encoder,
    tokenizer = paint_pipeline.tokenizer,
    unet = paint_pipeline.unet,
    scheduler = paint_pipeline.scheduler,
    safety_checker = paint_pipeline.safety_checker,
    feature_extractor = paint_pipeline.feature_extractor,
).to(device)