import os
import sys
import shutil
import torch
import fire
import gradio as gr
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
from functools import partial
import trimesh
import tempfile
from rembg import remove

code_dir = "../"
sys.path.append(code_dir)
from utils.zero123_utils import init_model, predict_stage1_gradio, zero123_infer
from utils.sam_utils import sam_init, sam_out_nosave
from utils.utils import image_preprocess_nosave, gen_poses
from elevation_estimate.estimate_wild_imgs import estimate_elev

_GPU_INDEX = 0
_HALF_PRECISION = True
_MESH_RESOLUTION = 256

_TITLE = '''One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization'''
_DESCRIPTION = '''
<div>
<a style="display:inline-block" href="http://one-2-3-45.com"><img src="https://img.shields.io/badge/Project_Homepage-f9f7f7?logo=data:image/webp;base64,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"></a>
<a style="display:inline-block; margin-left: .5em" href="https://arxiv.org/abs/2306.16928"><img src="https://img.shields.io/badge/2306.16928-f9f7f7?logo=data:image/png;base64,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"></a>
<a style="display:inline-block; margin-left: .5em" href='https://github.com/One-2-3-45/One-2-3-45'><img src='https://img.shields.io/github/stars/One-2-3-45/One-2-3-45?style=social' /></a>
</div>
We reconstruct a 3D textured mesh from a single image by initially predicting multi-view images and then lifting them to 3D. 
'''
_USER_GUIDE = "Please upload an image in the block above (or choose an example above) and click **Run Generation**." 
_BBOX_1 = "Predicting bounding box for the input image..."
_BBOX_2 = "Bounding box adjusted. Continue adjusting or **Run Generation**."
_BBOX_3 = "Bounding box predicted. Adjust it using sliders or **Run Generation**."
_SAM = "Preprocessing the input image... (safety check, SAM segmentation, *etc*.)"
_GEN_1 = "Predicting multi-view images... (may take \~13 seconds) <br> Images will be shown in the bottom right blocks."
_GEN_2 = "Predicting nearby views and generating mesh... (may take \~33 seconds) <br> Mesh will be shown on the right."
_DONE = "Done! Mesh is shown on the right. <br> If it is not satisfactory, please select **Retry view** checkboxes for inaccurate views and click **Regenerate selected view(s)** at the bottom."
_REGEN_1 = "Selected view(s) are regenerated. You can click **Regenerate nearby views and mesh**. <br> Alternatively, if the regenerated view(s) are still not satisfactory, you can repeat the previous step (select the view and regenerate)."
_REGEN_2 = "Regeneration done. Mesh is shown on the right."


def calc_cam_cone_pts_3d(polar_deg, azimuth_deg, radius_m, fov_deg):
    '''
    :param polar_deg (float).
    :param azimuth_deg (float).
    :param radius_m (float).
    :param fov_deg (float).
    :return (5, 3) array of float with (x, y, z).
    '''
    polar_rad = np.deg2rad(polar_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)
    fov_rad = np.deg2rad(fov_deg)
    polar_rad = -polar_rad  # NOTE: Inverse of how used_x relates to x.

    # Camera pose center:
    cam_x = radius_m * np.cos(azimuth_rad) * np.cos(polar_rad)
    cam_y = radius_m * np.sin(azimuth_rad) * np.cos(polar_rad)
    cam_z = radius_m * np.sin(polar_rad)

    # Obtain four corners of camera frustum, assuming it is looking at origin.
    # First, obtain camera extrinsics (rotation matrix only):
    camera_R = np.array([[np.cos(azimuth_rad) * np.cos(polar_rad),
                          -np.sin(azimuth_rad),
                          -np.cos(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(azimuth_rad) * np.cos(polar_rad),
                          np.cos(azimuth_rad),
                          -np.sin(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(polar_rad),
                          0.0,
                          np.cos(polar_rad)]])

    # Multiply by corners in camera space to obtain go to space:
    corn1 = [-1.0, np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn2 = [-1.0, -np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn3 = [-1.0, -np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn4 = [-1.0, np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn1 = np.dot(camera_R, corn1)
    corn2 = np.dot(camera_R, corn2)
    corn3 = np.dot(camera_R, corn3)
    corn4 = np.dot(camera_R, corn4)

    # Now attach as offset to actual 3D camera position:
    corn1 = np.array(corn1) / np.linalg.norm(corn1, ord=2)
    corn_x1 = cam_x + corn1[0]
    corn_y1 = cam_y + corn1[1]
    corn_z1 = cam_z + corn1[2]
    corn2 = np.array(corn2) / np.linalg.norm(corn2, ord=2)
    corn_x2 = cam_x + corn2[0]
    corn_y2 = cam_y + corn2[1]
    corn_z2 = cam_z + corn2[2]
    corn3 = np.array(corn3) / np.linalg.norm(corn3, ord=2)
    corn_x3 = cam_x + corn3[0]
    corn_y3 = cam_y + corn3[1]
    corn_z3 = cam_z + corn3[2]
    corn4 = np.array(corn4) / np.linalg.norm(corn4, ord=2)
    corn_x4 = cam_x + corn4[0]
    corn_y4 = cam_y + corn4[1]
    corn_z4 = cam_z + corn4[2]

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4]

    return np.array([xs, ys, zs]).T

class CameraVisualizer:
    def __init__(self, gradio_plot):
        self._gradio_plot = gradio_plot
        self._fig = None
        self._polar = 0.0
        self._azimuth = 0.0
        self._radius = 0.0
        self._raw_image = None
        self._8bit_image = None
        self._image_colorscale = None

    def encode_image(self, raw_image, elev=90):
        '''
        :param raw_image (H, W, 3) array of uint8 in [0, 255].
        '''
        # https://stackoverflow.com/questions/60685749/python-plotly-how-to-add-an-image-to-a-3d-scatter-plot

        dum_img = Image.fromarray(np.ones((3, 3, 3), dtype='uint8')).convert('P', palette='WEB')
        idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))

        self._raw_image = raw_image
        self._8bit_image = Image.fromarray(raw_image).convert('P', palette='WEB', dither=None)
        # self._8bit_image = Image.fromarray(raw_image.clip(0, 254)).convert(
        #     'P', palette='WEB', dither=None)
        self._image_colorscale = [
            [i / 255.0, 'rgb({}, {}, {})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]
        self._elev = elev
        # return self.update_figure()

    def update_figure(self):
        fig = go.Figure()

        if self._raw_image is not None:
            (H, W, C) = self._raw_image.shape

            x = np.zeros((H, W))
            (y, z) = np.meshgrid(np.linspace(-1.0, 1.0, W), np.linspace(1.0, -1.0, H) * H / W)
            
            angle_deg = self._elev-90
            angle = np.radians(90-self._elev)
            rotation_matrix = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
            # Assuming x, y, z are the original 3D coordinates of the image
            coordinates = np.stack((x, y, z), axis=-1)  # Combine x, y, z into a single array
            # Apply the rotation matrix
            rotated_coordinates = np.matmul(coordinates, rotation_matrix)
            # Extract the new x, y, z coordinates from the rotated coordinates
            x, y, z = rotated_coordinates[..., 0], rotated_coordinates[..., 1], rotated_coordinates[..., 2]

            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                surfacecolor=self._8bit_image,
                cmin=0,
                cmax=255,
                colorscale=self._image_colorscale,
                showscale=False,
                lighting_diffuse=1.0,
                lighting_ambient=1.0,
                lighting_fresnel=1.0,
                lighting_roughness=1.0,
                lighting_specular=0.3))

            scene_bounds = 3.5
            base_radius = 2.5
            zoom_scale = 1.5  # Note that input radius offset is in [-0.5, 0.5].
            fov_deg = 50.0
            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]

            input_cone = calc_cam_cone_pts_3d(
                angle_deg, 0.0, base_radius, fov_deg)  # (5, 3).
            output_cone = calc_cam_cone_pts_3d(
                self._polar, self._azimuth, base_radius + self._radius * zoom_scale, fov_deg)  # (5, 3).
            output_cones = []
            for i in range(1,4):
                output_cones.append(calc_cam_cone_pts_3d(
                    angle_deg, i*90, base_radius + self._radius * zoom_scale, fov_deg))
            delta_deg = 30 if angle_deg <= -15 else -30
            for i in range(4):
                output_cones.append(calc_cam_cone_pts_3d(
                    angle_deg+delta_deg, 30+i*90, base_radius + self._radius * zoom_scale, fov_deg))

            cones = [(input_cone, 'rgb(174, 54, 75)', 'Input view (Predicted view 1)')]
            for i in range(len(output_cones)):
                cones.append((output_cones[i], 'rgb(32, 77, 125)', f'Predicted view {i+2}'))

            for idx, (cone, clr, legend) in enumerate(cones):

                for (i, edge) in enumerate(edges):
                    (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
                    (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
                    (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
                    fig.add_trace(go.Scatter3d(
                        x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                        line=dict(color=clr, width=3),
                        name=legend, showlegend=(i == 1) and (idx <= 1)))

                # Add label.
                if cone[0, 2] <= base_radius / 2.0:
                    fig.add_trace(go.Scatter3d(
                        x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] - 0.05], showlegend=False,
                        mode='text', text=legend, textposition='bottom center'))
                else:
                    fig.add_trace(go.Scatter3d(
                        x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] + 0.05], showlegend=False,
                        mode='text', text=legend, textposition='top center'))

            # look at center of scene
            fig.update_layout(
                height=450,
                autosize=True,
                hovermode=False,
                margin=go.layout.Margin(l=0, r=0, b=0, t=0),
                showlegend=False,
                legend=dict(
                    yanchor='bottom',
                    y=0.01,
                    xanchor='right',
                    x=0.99,
                ),
                scene=dict(
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=1.0),
                    camera=dict(
                        eye=dict(x=base_radius - 1.6, y=0.0, z=0.6),
                        center=dict(x=0.0, y=0.0, z=0.0),
                        up=dict(x=0.0, y=0.0, z=1.0)),
                    xaxis_title='',
                    yaxis_title='',
                    zaxis_title='',
                    xaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks=''),
                    yaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks=''),
                    zaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks='')))

        self._fig = fig
        return fig
    

def stage1_run(models, device, cam_vis, tmp_dir,
               input_im, scale, ddim_steps, elev=None, rerun_all=[],
               *btn_retrys):
    is_rerun = True if cam_vis is None else False
    model = models['turncam']

    stage1_dir = os.path.join(tmp_dir, "stage1_8")
    if not is_rerun:
        os.makedirs(stage1_dir, exist_ok=True)
        output_ims = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4)), device=device, ddim_steps=ddim_steps, scale=scale)
        stage2_steps = 50 # ddim_steps
        zero123_infer(model, tmp_dir, indices=[0], device=device, ddim_steps=stage2_steps, scale=scale)
        try:
            elev_output = int(estimate_elev(tmp_dir))
        except:
            print("Failed to estimate polar angle")
            elev_output = 90
        print("Estimated polar angle:", elev_output)
        gen_poses(tmp_dir, elev_output)
        show_in_im1 = np.asarray(input_im, dtype=np.uint8)
        cam_vis.encode_image(show_in_im1, elev=elev_output)
        new_fig = cam_vis.update_figure()

        flag_lower_cam = elev_output <= 75
        if flag_lower_cam:
            output_ims_2 = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4,8)), device=device, ddim_steps=ddim_steps, scale=scale)
        else:
            output_ims_2 = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(8,12)), device=device, ddim_steps=ddim_steps, scale=scale)
        torch.cuda.empty_cache()
        return (90-elev_output, new_fig, *output_ims, *output_ims_2)
    else:
        rerun_idx = [i for i in range(len(btn_retrys)) if btn_retrys[i]]
        if 90-int(elev["label"]) > 75:
            rerun_idx_in = [i if i < 4 else i+4 for i in rerun_idx]
        else:
            rerun_idx_in = rerun_idx
        for idx in rerun_idx_in:
            if idx not in rerun_all:
                rerun_all.append(idx)
        print("rerun_idx", rerun_all)
        output_ims = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=rerun_idx_in, device=device, ddim_steps=ddim_steps, scale=scale)
        outputs = [gr.update(visible=True)] * 8
        for idx, view_idx in enumerate(rerun_idx):
            outputs[view_idx] = output_ims[idx]
        reset = [gr.update(value=False)] * 8
        torch.cuda.empty_cache()
        return (rerun_all, *reset, *outputs)
    
def stage2_run(models, device, tmp_dir,
               elev, scale, is_glb=False, rerun_all=[], stage2_steps=50):
    flag_lower_cam = 90-int(elev["label"]) <= 75
    is_rerun = True if rerun_all else False
    model = models['turncam']
    if not is_rerun:
        if flag_lower_cam:
            zero123_infer(model, tmp_dir, indices=list(range(1,8)), device=device, ddim_steps=stage2_steps, scale=scale)
        else:
            zero123_infer(model, tmp_dir, indices=list(range(1,4))+list(range(8,12)), device=device, ddim_steps=stage2_steps, scale=scale)
    else:
        print("rerun_idx", rerun_all)
        zero123_infer(model, tmp_dir, indices=rerun_all, device=device, ddim_steps=stage2_steps, scale=scale)
    
    dataset = tmp_dir
    main_dir_path = os.path.dirname(__file__)
    torch.cuda.empty_cache()
    os.chdir(os.path.join(code_dir, 'reconstruction/'))

    bash_script = f'CUDA_VISIBLE_DEVICES={_GPU_INDEX} python exp_runner_generic_blender_val.py \
                    --specific_dataset_name {dataset} \
                    --mode export_mesh \
                    --conf confs/one2345_lod0_val_demo.conf \
                    --resolution {_MESH_RESOLUTION}'
    print(bash_script)
    os.system(bash_script)
    os.chdir(main_dir_path)

    ply_path = os.path.join(tmp_dir, f"mesh.ply")
    mesh_ext = ".glb" if is_glb else ".obj"
    mesh_path = os.path.join(tmp_dir, f"mesh{mesh_ext}")
    # Read the textured mesh from .ply file
    mesh = trimesh.load_mesh(ply_path)
    rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    mesh.apply_transform(rotation_matrix)
    rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
    mesh.apply_transform(rotation_matrix)
    # flip x
    mesh.vertices[:, 0] = -mesh.vertices[:, 0]
    mesh.faces = np.fliplr(mesh.faces)
    # Export the mesh as .obj file with colors
    if not is_glb:
        mesh.export(mesh_path, file_type='obj', include_color=True)
    else:
        mesh.export(mesh_path, file_type='glb')
    torch.cuda.empty_cache()

    if not is_rerun:
        return (mesh_path)
    else:
        return (mesh_path, gr.update(value=[]), gr.update(visible=False), gr.update(visible=False))

def nsfw_check(models, raw_im, device='cuda'):
    safety_checker_input = models['clip_fe'](raw_im, return_tensors='pt').to(device)
    (_, has_nsfw_concept) = models['nsfw'](
        images=np.ones((1, 3)), clip_input=safety_checker_input.pixel_values)
    del safety_checker_input
    if np.any(has_nsfw_concept):
        print('NSFW content detected.')
        return Image.open("unsafe.png")
    else:
        print('Safety check passed.')
        return False

def preprocess_run(predictor, models, raw_im, lower_contrast, *bbox_sliders):
    raw_im.thumbnail([512, 512], Image.Resampling.LANCZOS)
    check_results = nsfw_check(models, raw_im, device=predictor.device)
    if check_results:
        return check_results
    image_sam = sam_out_nosave(predictor, raw_im.convert("RGB"), *bbox_sliders)
    input_256 = image_preprocess_nosave(image_sam, lower_contrast=lower_contrast, rescale=True)
    torch.cuda.empty_cache()
    return input_256

def on_coords_slider(image, x_min, y_min, x_max, y_max, color=(88, 191, 131, 255)):
    """Draw a bounding box annotation for an image."""
    print("Slider adjusted, drawing bbox...")
    image.thumbnail([512, 512], Image.Resampling.LANCZOS)
    image_size = image.size
    if max(image_size) > 224:
        image.thumbnail([224, 224], Image.Resampling.LANCZOS)
        shrink_ratio = max(image.size) / max(image_size)
        x_min = int(x_min * shrink_ratio)
        y_min = int(y_min * shrink_ratio)
        x_max = int(x_max * shrink_ratio)
        y_max = int(y_max * shrink_ratio)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, int(max(max(image.shape) / 400*2, 2)))
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA) # image[:, :, ::-1]

def init_bbox(image):
    image.thumbnail([512, 512], Image.Resampling.LANCZOS)
    width, height = image.size
    image_rem = image.convert('RGBA')
    image_nobg = remove(image_rem, alpha_matting=True)
    arr = np.asarray(image_nobg)[:,:,-1]
    x_nonzero = np.nonzero(arr.sum(axis=0))
    y_nonzero = np.nonzero(arr.sum(axis=1))
    x_min = int(x_nonzero[0].min())
    y_min = int(y_nonzero[0].min())
    x_max = int(x_nonzero[0].max())
    y_max = int(y_nonzero[0].max())
    image_mini = image.copy()
    image_mini.thumbnail([224, 224], Image.Resampling.LANCZOS)
    shrink_ratio = max(image_mini.size) / max(width, height)
    x_min_shrink = int(x_min * shrink_ratio)
    y_min_shrink = int(y_min * shrink_ratio)
    x_max_shrink = int(x_max * shrink_ratio)
    y_max_shrink = int(y_max * shrink_ratio)

    return [on_coords_slider(image_mini, x_min_shrink, y_min_shrink, x_max_shrink, y_max_shrink),
            gr.update(value=x_min, maximum=width),
            gr.update(value=y_min, maximum=height),
            gr.update(value=x_max, maximum=width),
            gr.update(value=y_max, maximum=height)]


def run_demo(
        device_idx=_GPU_INDEX,
        ckpt='zero123-xl.ckpt'):

    device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
    models = init_model(device, os.path.join(code_dir, 'zero123-xl.ckpt'), half_precision=_HALF_PRECISION)

    # init sam model
    predictor = sam_init(device_idx)

    with open('instructions_12345.md', 'r') as f:
        article = f.read()

    # NOTE: Examples must match inputs
    example_folder = os.path.join(os.path.dirname(__file__), 'demo_examples')
    example_fns = os.listdir(example_folder)
    example_fns.sort()
    examples_full = [os.path.join(example_folder, x) for x in example_fns if x.endswith('.png')]

    # Compose demo layout & data flow.
    with gr.Blocks(title=_TITLE, css="style.css") as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('# ' + _TITLE)
            with gr.Column(scale=0):
                gr.DuplicateButton(value='Duplicate Space for private use',
                            elem_id='duplicate-button')
        gr.Markdown(_DESCRIPTION)

        with gr.Row(variant='panel'):
            with gr.Column(scale=6):
                image_block = gr.Image(type='pil', image_mode='RGBA', height=290, label='Input image', tool=None)

                gr.Examples(
                    examples=examples_full,  # NOTE: elements must match inputs list!
                    inputs=[image_block],
                    outputs=[image_block],
                    cache_examples=False,
                    label='Examples (click one of the images below to start)',
                    examples_per_page=40
                )
                preprocess_chk = gr.Checkbox(
                        False, label='Reduce image contrast (mitigate shadows on the backside)')
                with gr.Accordion('Advanced options', open=False):
                    scale_slider = gr.Slider(0, 30, value=3, step=1,
                                             label='Diffusion guidance scale')
                    steps_slider = gr.Slider(5, 200, value=75, step=5,
                                             label='Number of diffusion inference steps')
                    glb_chk = gr.Checkbox(
                        False, label='Export the mesh in .glb format')

                run_btn = gr.Button('Run Generation', variant='primary', interactive=False)
                guide_text = gr.Markdown(_USER_GUIDE, visible=True)
                
            with gr.Column(scale=4):
                with gr.Row():
                    bbox_block = gr.Image(type='pil', label="Bounding box", height=290, interactive=False)
                    sam_block = gr.Image(type='pil', label="SAM output", interactive=False)
                max_width = max_height = 256
                with gr.Row():
                    x_min_slider = gr.Slider(label="X min", interactive=True, value=0, minimum=0, maximum=max_width, step=1)
                    y_min_slider = gr.Slider(label="Y min", interactive=True, value=0, minimum=0, maximum=max_height, step=1)
                with gr.Row():
                    x_max_slider = gr.Slider(label="X max", interactive=True, value=max_width, minimum=0, maximum=max_width, step=1)
                    y_max_slider = gr.Slider(label="Y max", interactive=True, value=max_height, minimum=0, maximum=max_height, step=1)
                bbox_sliders = [x_min_slider, y_min_slider, x_max_slider, y_max_slider]

                mesh_output = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="One-2-3-45's Textured Mesh", elem_id="model-3d-out")
        
        with gr.Row(variant='panel'):
            with gr.Column(scale=85):
                elev_output = gr.Label(label='Estimated elevation (degree, w.r.t. the horizontal plane)')
                vis_output = gr.Plot(label='Camera poses of the input view (red) and predicted views (blue)', elem_id="plot-out")
                
            with gr.Column(scale=115):
                gr.Markdown('Predicted multi-view images')
                with gr.Row():
                    view_1 = gr.Image(interactive=False, height=200, show_label=False)
                    view_2 = gr.Image(interactive=False, height=200, show_label=False)
                    view_3 = gr.Image(interactive=False, height=200, show_label=False)
                    view_4 = gr.Image(interactive=False, height=200, show_label=False)
                with gr.Row():
                    btn_retry_1 = gr.Checkbox(label='Retry view 1')
                    btn_retry_2 = gr.Checkbox(label='Retry view 2')
                    btn_retry_3 = gr.Checkbox(label='Retry view 3')
                    btn_retry_4 = gr.Checkbox(label='Retry view 4')
                with gr.Row():
                    view_5 = gr.Image(interactive=False, height=200, show_label=False)
                    view_6 = gr.Image(interactive=False, height=200, show_label=False)
                    view_7 = gr.Image(interactive=False, height=200, show_label=False)
                    view_8 = gr.Image(interactive=False, height=200, show_label=False)
                with gr.Row():
                    btn_retry_5 = gr.Checkbox(label='Retry view 5')
                    btn_retry_6 = gr.Checkbox(label='Retry view 6')
                    btn_retry_7 = gr.Checkbox(label='Retry view 7')
                    btn_retry_8 = gr.Checkbox(label='Retry view 8')
                with gr.Row():
                    regen_view_btn = gr.Button('1. Regenerate selected view(s)', variant='secondary', visible=False)
                    regen_mesh_btn = gr.Button('2. Regenerate nearby views and mesh', variant='secondary', visible=False)
        
        gr.Markdown(article)
        gr.HTML("""
            <div class="footer">
                <p> 
                One-2-3-45 Demo by <a style="text-decoration:none" href="https://chaoxu.xyz" target="_blank">Chao Xu</a>
                </p>
            </div>
        """)

        update_guide = lambda GUIDE_TEXT: gr.update(value=GUIDE_TEXT)

        views = [view_1, view_2, view_3, view_4, view_5, view_6, view_7, view_8]
        btn_retrys = [btn_retry_1, btn_retry_2, btn_retry_3, btn_retry_4, btn_retry_5, btn_retry_6, btn_retry_7, btn_retry_8]
        
        rerun_idx = gr.State([])
        tmp_dir = gr.State('./demo_tmp/tmp_dir')

        def refresh(tmp_dir):
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            tmp_dir = tempfile.TemporaryDirectory(dir=os.path.join(os.path.dirname(__file__), 'demo_tmp'))
            print("create tmp_dir", tmp_dir.name)
            clear = [gr.update(value=[])] + [None] * 5 + [gr.update(visible=False)] * 2 + [None] * 8 + [gr.update(value=False)] * 8
            return (tmp_dir.name, *clear)
        
        placeholder = gr.Image(visible=False)
        tmp_func = lambda x: False if not x else gr.update(visible=False)
        disable_func = lambda x: gr.update(interactive=False)
        enable_func = lambda x: gr.update(interactive=True)
        image_block.change(disable_func, inputs=run_btn, outputs=run_btn, queue=False
                           ).success(fn=refresh,
                           inputs=[tmp_dir],
                           outputs=[tmp_dir, rerun_idx, bbox_block, sam_block, elev_output, vis_output, mesh_output, regen_view_btn, regen_mesh_btn, *views, *btn_retrys],
                           queue=False
                           ).success(fn=tmp_func, inputs=[image_block], outputs=[placeholder], queue=False
                           ).success(fn=partial(update_guide, _BBOX_1), outputs=[guide_text], queue=False
                           ).success(fn=init_bbox,
                                     inputs=[image_block],
                                     outputs=[bbox_block, *bbox_sliders], queue=False
                           ).success(fn=partial(update_guide, _BBOX_3), outputs=[guide_text], queue=False
                           ).success(enable_func, inputs=run_btn, outputs=run_btn, queue=False)


        for bbox_slider in bbox_sliders:
            bbox_slider.release(fn=on_coords_slider,
                               inputs=[image_block, *bbox_sliders],
                               outputs=[bbox_block], 
                               queue=False
                               ).success(fn=partial(update_guide, _BBOX_2), outputs=[guide_text], queue=False)

        cam_vis = CameraVisualizer(vis_output)

        # Define the function to be called when any of the btn_retry buttons are clicked
        def on_retry_button_click(*btn_retrys):
            any_checked = any([btn_retry for btn_retry in btn_retrys])
            print('any_checked:', any_checked, [btn_retry for btn_retry in btn_retrys])
            if any_checked:
                return (gr.update(visible=True), gr.update(visible=True))
            else:
                return (gr.update(), gr.update())
        # make regen_btn visible when any of the btn_retry is checked
        for btn_retry in btn_retrys:
            # Add the event handlers to the btn_retry buttons
            btn_retry.change(fn=on_retry_button_click, inputs=[*btn_retrys], outputs=[regen_view_btn, regen_mesh_btn], queue=False)


        run_btn.click(fn=partial(update_guide, _SAM), outputs=[guide_text], queue=False
                      ).success(fn=partial(preprocess_run, predictor, models), 
                                inputs=[image_block, preprocess_chk, *bbox_sliders], 
                                outputs=[sam_block]
                      ).success(fn=partial(update_guide, _GEN_1), outputs=[guide_text], queue=False
                      ).success(fn=partial(stage1_run, models, device, cam_vis),
                                inputs=[tmp_dir, sam_block, scale_slider, steps_slider],
                                outputs=[elev_output, vis_output, *views]
                      ).success(fn=partial(update_guide, _GEN_2), outputs=[guide_text], queue=False
                      ).success(fn=partial(stage2_run, models, device),
                                inputs=[tmp_dir, elev_output, scale_slider, glb_chk],
                                outputs=[mesh_output]
                      ).success(fn=partial(update_guide, _DONE), outputs=[guide_text], queue=False)
    

        regen_view_btn.click(fn=partial(stage1_run, models, device, None),
                             inputs=[tmp_dir, sam_block, scale_slider, steps_slider, elev_output, rerun_idx, *btn_retrys],
                             outputs=[rerun_idx, *btn_retrys, *views]
                            ).success(fn=partial(update_guide, _REGEN_1), outputs=[guide_text], queue=False)
        regen_mesh_btn.click(fn=partial(stage2_run, models, device),
                             inputs=[tmp_dir, elev_output, scale_slider, glb_chk, rerun_idx],
                             outputs=[mesh_output, rerun_idx, regen_view_btn, regen_mesh_btn]
                            ).success(fn=partial(update_guide, _REGEN_2), outputs=[guide_text], queue=False)


    demo.queue().launch(share=True, max_threads=80) # auth=("admin", os.environ['PASSWD'])


if __name__ == '__main__':
    fire.Fire(run_demo)
