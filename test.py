from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import json
import os

device = "cuda"
# use DDIM scheduler, you can modify it to use other scheduler
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=True)

# modify the model path
pipe = StableDiffusionPipeline.from_pretrained(
    f"./output-models/1500/",
    scheduler=scheduler,
    safety_checker=None,
    torch_dtype=torch.float16,
).to(device)

# enable xformers memory attention
pipe.enable_xformers_memory_efficient_attention()

# Process for bulk testing
concepts_list = "/media/aioz-thang/data3/aioz-tuongdo/tune1.0/Test/concepts_list.json"
with open(concepts_list, "r") as f:
            concepts_list = json.load(f)
prompts = []
for concept in concepts_list:
            prompt = concept["instance_prompt"]
            print(prompt)
            prompts.append(prompt)
print(len(prompts))

# Process for demo
#prompt = "This lady looks serious without smile in the face."

negative_prompt = ""
num_samples = 4
guidance_scale = 7.5
num_inference_steps = 20
height = 512
width = 512

# Test
index = 1
for prompt in prompts:
    with torch.autocast("cuda"), torch.inference_mode():
        images = pipe(prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
        ).images

    count = 1
    for image in images:
        # save image to local directory
        directory = "output/"+str(index)+"/"
        if (os.path.isdir(directory) is False):
            os.mkdir(directory)
        image.save("output/" + str(index) + "/" + str(count)+".png")
        count += 1
    index += 1
