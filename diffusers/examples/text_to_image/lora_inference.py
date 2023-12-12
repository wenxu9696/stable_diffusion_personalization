import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import numpy as np

model_base = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16, use_safetensors=True)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

lora_model_path = 'output/lora/pokemon/pytorch_lora_weights.safetensors'
pipe.unet.load_attn_procs(lora_model_path)
pipe.to("mps")


text_caption = 'A yellow pokemon with a sword'
filename = text_caption.replace(' ', '_')
print(filename)
images_0 = pipe(text_caption, num_inference_steps=25, guidance_scale=7.0, cross_attention_kwargs={"scale": 0.0}).images
image_0 = images_0[0]
image_0.save(f"{filename}_0.png")


# use half the weights from the LoRA finetuned model and half the weights from the base model
images_05 = pipe(text_caption, num_inference_steps=25, guidance_scale=7.0, cross_attention_kwargs={"scale": 0.5}).images
image_05 = images_05[0]
image_05.save(f"{filename}_0.5.png")

# use the weights from the fully finetuned LoRA model
images_1 = pipe(text_caption, num_inference_steps=25, guidance_scale=7.0).images
image_1 = images_1[0]
image_1.save(f"{filename}_1.png")


clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)


sd_clip_score = calculate_clip_score(np.array(image_1)[None, :], [text_caption,])
print(f"{text_caption} weight 1, CLIP score: {sd_clip_score}")

sd_clip_score = calculate_clip_score(np.array(image_0)[None, :], [text_caption,])
print(f"{text_caption} weight 0, CLIP score: {sd_clip_score}")

sd_clip_score = calculate_clip_score(np.array(image_05)[None, :], [text_caption,])
print(f"{text_caption} weight 0.5, CLIP score: {sd_clip_score}")
