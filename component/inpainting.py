import numpy as np
from PIL import Image
import torch
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
)


class ControlNetInpainter:
    def __init__(self):
        self.controlnet = ControlNetModel.from_pretrained(
            "model/controlnet-depth-sdxl-1.0",
            torch_dtype=torch.float32
        )
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "model/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            torch_dtype=torch.float32,
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe_ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "model/ip2p",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        ).to("cuda")
        self.generator = torch.manual_seed(0)

    def pre_process(self, image, image_control, image_mask):
        image = Image.fromarray(image)
        image_control = Image.fromarray(image_control)
        image_mask = Image.fromarray(image_mask)

        if image.mode == "RGBA":
            image = image.convert("RGB")
        original_width, original_height = image.size
        new_width = int(original_width / 1.5)
        new_height = int(original_height / 1.5)
        image = image.resize((new_width, new_height))
        image_control = image_control.resize((new_width, new_height))
        image_mask = image_mask.resize((new_width, new_height))

        return image, image_mask, image_control

    def inpainting(self, image, image_mask, image_control, template_name):
        image, image_mask, image_control = self.pre_process(image, image_control, image_mask)
        prompt = f"{template_name}, hand, realskin, photorealistic, RAW photo, best quality, realistic, photo-realistic, best quality ,masterpiece,an extremely delicate and beautiful, extremely detailed,2k wallpaper, Amazing, finely detail, 8k wallpaper,huge filesize , ultra-detailed, highres, extremely detailed,realistic"
        negative_prompt = "deformed, EasyNegative, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)),"

        image_inpainting = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=image_mask,
            control_image=image_control,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            generator=self.generator,
            controlnet_conditioning_scale=1.0,
            strength=0.75,
            guidance_scale=9.0,
        )[0][0]

        return np.array(image_inpainting)

    def ip2p(self, image_inpainting):
        image_inpainting = Image.fromarray(image_inpainting)

        # Further process the output using self.pipe_ip2p
        image_ip2p = self.pipe_ip2p(
            "Turn the deformed hand into normal",
            image=image_inpainting,
            num_inference_steps=20,
            image_guidance_scale=1.5,
            guidance_scale=10,
        )[0][0]

        return np.array(image_ip2p)
