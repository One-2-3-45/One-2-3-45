import os
import numpy as np
import torch
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from transformers import CLIPImageProcessor
from torch import autocast
from torchvision import transforms


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


def init_model(device, ckpt, half_precision=False):
    config = os.path.join(os.path.dirname(__file__), '../configs/sd-objaverse-finetune-c_concat-256.yaml')
    config = OmegaConf.load(config)

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print('Instantiating LatentDiffusion...')
    if half_precision:
        models['turncam'] = torch.compile(load_model_from_config(config, ckpt, device=device)).half()
    else:
        models['turncam'] = torch.compile(load_model_from_config(config, ckpt, device=device))
    print('Instantiating StableDiffusionSafetyChecker...')
    models['nsfw'] = StableDiffusionSafetyChecker.from_pretrained(
        'CompVis/stable-diffusion-safety-checker').to(device)
    models['clip_fe'] = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14")
    # We multiply all by some factor > 1 to make them less likely to be triggered.
    models['nsfw'].concept_embeds_weights *= 1.2
    models['nsfw'].special_care_embeds_weights *= 1.2

    return models

@torch.no_grad()
def sample_model_batch(model, sampler, input_im, xs, ys, n_samples=4, precision='autocast', ddim_eta=1.0, ddim_steps=75, scale=3.0, h=256, w=256):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = []
            for x, y in zip(xs, ys):
                T.append([np.radians(x), np.sin(np.radians(y)), np.cos(np.radians(y)), 0])
            T = torch.tensor(np.array(T))[:, None, :].float().to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage(input_im).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            # print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            ret_imgs = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
            del cond, c, x_samples_ddim, samples_ddim, uc, input_im
            torch.cuda.empty_cache()
            return ret_imgs

@torch.no_grad()
def predict_stage1_gradio(model, raw_im, save_path = "", adjust_set=[], device="cuda", ddim_steps=75, scale=3.0):
    # raw_im = raw_im.resize([256, 256], Image.LANCZOS)
    # input_im_init = preprocess_image(models, raw_im, preprocess=False)
    input_im_init = np.asarray(raw_im, dtype=np.float32) / 255.0
    input_im = transforms.ToTensor()(input_im_init).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1

    # stage 1: 8
    delta_x_1_8 = [0] * 4 + [30] * 4 + [-30] * 4
    delta_y_1_8 = [0+90*(i%4) if i < 4 else 30+90*(i%4) for i in range(8)] + [30+90*(i%4) for i in range(4)]

    ret_imgs = []
    sampler = DDIMSampler(model)
    # sampler.to(device)
    if adjust_set != []:
        x_samples_ddims_8 = sample_model_batch(model, sampler, input_im, 
                                               [delta_x_1_8[i] for i in adjust_set], [delta_y_1_8[i] for i in adjust_set], 
                                               n_samples=len(adjust_set), ddim_steps=ddim_steps, scale=scale)
    else:
        x_samples_ddims_8 = sample_model_batch(model, sampler, input_im, delta_x_1_8, delta_y_1_8, n_samples=len(delta_x_1_8), ddim_steps=ddim_steps, scale=scale)
    sample_idx = 0
    for stage1_idx in range(len(delta_x_1_8)):
        if adjust_set != [] and stage1_idx not in adjust_set:
            continue
        x_sample = 255.0 * rearrange(x_samples_ddims_8[sample_idx].numpy(), 'c h w -> h w c')
        out_image = Image.fromarray(x_sample.astype(np.uint8))
        ret_imgs.append(out_image)
        if save_path:
            out_image.save(os.path.join(save_path, '%d.png'%(stage1_idx)))
        sample_idx += 1
    del x_samples_ddims_8
    del sampler
    torch.cuda.empty_cache()
    return ret_imgs

def infer_stage_2(model, save_path_stage1, save_path_stage2, delta_x_2, delta_y_2, indices, device, ddim_steps=75, scale=3.0):
    for stage1_idx in indices:
        # save stage 1 image
        # x_sample = 255.0 * rearrange(x_samples_ddims[stage1_idx].cpu().numpy(), 'c h w -> h w c')
        # Image.fromarray(x_sample.astype(np.uint8)).save()
        stage1_image_path = os.path.join(save_path_stage1, '%d.png'%(stage1_idx))

        raw_im = Image.open(stage1_image_path)
        # input_im_init = preprocess_image(models, raw_im, preprocess=False)
        input_im_init = np.asarray(raw_im, dtype=np.float32) #/ 255.0
        input_im_init[input_im_init >= 253.0] = 255.0
        input_im_init = input_im_init / 255.0
        input_im = transforms.ToTensor()(input_im_init).unsqueeze(0).to(device)
        input_im = input_im * 2 - 1
        # infer stage 2
        sampler = DDIMSampler(model)
        # sampler.to(device)
        # stage2_in = x_samples_ddims[stage1_idx][None, ...].to(device) * 2 - 1
        x_samples_ddims_stage2 = sample_model_batch(model, sampler, input_im, delta_x_2, delta_y_2, n_samples=len(delta_x_2), ddim_steps=ddim_steps, scale=scale)
        for stage2_idx in range(len(delta_x_2)):
            x_sample_stage2 = 255.0 * rearrange(x_samples_ddims_stage2[stage2_idx].numpy(), 'c h w -> h w c')
            Image.fromarray(x_sample_stage2.astype(np.uint8)).save(os.path.join(save_path_stage2, '%d_%d.png'%(stage1_idx, stage2_idx)))
        del input_im
        del x_samples_ddims_stage2
        torch.cuda.empty_cache()

def zero123_infer(model, input_dir_path, start_idx=0, end_idx=12, indices=None, device="cuda", ddim_steps=75, scale=3.0):
    # input_img_path = os.path.join(input_dir_path, "input_256.png")
    save_path_8 = os.path.join(input_dir_path, "stage1_8")
    save_path_8_2 = os.path.join(input_dir_path, "stage2_8")
    os.makedirs(save_path_8_2, exist_ok=True)

    # raw_im = Image.open(input_img_path)
    # # input_im_init = preprocess_image(models, raw_im, preprocess=False)
    # input_im_init = np.asarray(raw_im, dtype=np.float32) / 255.0
    # input_im = transforms.ToTensor()(input_im_init).unsqueeze(0).to(device)
    # input_im = input_im * 2 - 1

    # stage 2: 6*4 or 8*4
    delta_x_2 = [-10, 10, 0, 0]
    delta_y_2 = [0, 0, -10, 10]
    
    infer_stage_2(model, save_path_8, save_path_8_2, delta_x_2, delta_y_2, indices=indices if indices else list(range(start_idx,end_idx)), device=device, ddim_steps=ddim_steps, scale=scale)
