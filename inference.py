import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import imageio
from pathlib import Path
from PIL import Image
from amodal3r.pipelines import Amodal3RImageTo3DPipeline
from amodal3r.utils import render_utils, postprocessing_utils
import cv2
import trimesh

scratch = os.environ["SCRATCH"]


def prepare_masks_and_image(
    target_mask_path,
    occ_mask_path,
    image_path,
    canvas_size=518,
    fit="downscale",
    margin=0,
    bg=255,
    resize_image_if_needed=True,
):
    target = cv2.imread(str(target_mask_path), cv2.IMREAD_GRAYSCALE)
    occ = cv2.imread(str(occ_mask_path), cv2.IMREAD_GRAYSCALE)
    if target is None or occ is None:
        raise FileNotFoundError("Could not read target or occluder mask")

    target = target > 0
    occ = occ > 0

    Hm, Wm = target.shape

    union = target | occ
    if not union.any():
        blank = np.full((canvas_size, canvas_size, 3), bg, np.uint8)
        return blank, None, (target, occ), {
            "scale": 1.0, "center": (0, 0), "bbox": (0, -1, 0, -1),
            "M": np.eye(2, 3, dtype=np.float32), "margin": margin
        }

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Couldn't read image: {image_path}")
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    Hi, Wi = image.shape[:2]
    if (Hi, Wi) != (Hm, Wm) and resize_image_if_needed:
        image = cv2.resize(image, (Wm, Hm), interpolation=cv2.INTER_LINEAR)

    ys, xs = np.where(union)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h, w = (y1 - y0 + 1), (x1 - x0 + 1)
    cy = (y0 + y1) / 2.0
    cx = (x0 + x1) / 2.0

    avail = max(1, canvas_size - 2 * margin)
    s_fit = min(avail / h, avail / w)
    if fit == "none":
        s = 1.0
    elif fit == "downscale":
        s = min(1.0, s_fit)
    else:
        s = s_fit

    tx = canvas_size / 2.0 - s * cx
    ty = canvas_size / 2.0 - s * cy
    M = np.array([[s, 0, tx], [0, s, ty]], dtype=np.float32)

    warped_img = cv2.warpAffine(
        image, M, (canvas_size, canvas_size),
        flags=cv2.INTER_LINEAR, borderValue=(bg, bg, bg)
    )

    t_u8 = (target.astype(np.uint8) * 255)
    o_u8 = (occ.astype(np.uint8) * 255)

    target_w = cv2.warpAffine(
        t_u8, M, (canvas_size, canvas_size),
        flags=cv2.INTER_NEAREST, borderValue=0
    ) > 0

    occ_w = cv2.warpAffine(
        o_u8, M, (canvas_size, canvas_size),
        flags=cv2.INTER_NEAREST, borderValue=0
    ) > 0

    canvas = np.full((canvas_size, canvas_size), 255, np.uint8)
    canvas[target_w] = np.uint8(188)
    canvas[occ_w] = np.uint8(0)


    #converting to PIL Image for compatibility
    canvas = Image.fromarray(canvas, mode="L")
    warped_img = Image.fromarray(warped_img)

    return canvas, warped_img





def extract_glb(gs, mesh, mesh_simplify=0.95, texture_size=1024, export_path="output.glb"):
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        mesh_simplify (float): The mesh simplification factor.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """

    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb.export(export_path)
    return export_path

def save_mesh(mesh_result, filename):
    vertices = mesh_result.vertices.cpu().numpy() if hasattr(mesh_result.vertices, 'cpu') else mesh_result.vertices
    faces = mesh_result.faces.cpu().numpy() if hasattr(mesh_result.faces, 'cpu') else mesh_result.faces
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    if mesh_result.vertex_attrs is not None:
        attrs = mesh_result.vertex_attrs.cpu().numpy() if hasattr(mesh_result.vertex_attrs, 'cpu') else mesh_result.vertex_attrs
        mesh.visual.vertex_colors = attrs
    
    mesh.export(filename)

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = Amodal3RImageTo3DPipeline.from_pretrained("Sm0kyWu/Amodal3R")
pipeline.cuda()


output_dir = "./output/1/"
os.makedirs(output_dir, exist_ok=True)

unc_image_dir = Path(f"{scratch}/room/output/threedfront/scene/unc_images")
room = "Bedroom/Bedroom-1573"
room_folder = unc_image_dir / room

target_mask_path = room_folder / "000_mask_furniture_5_ddef48db-000d-4614-94c4-67ce114bb0e9_0.png"
occ_mask_path = room_folder / "000_occlusion_furniture_5_ddef48db-000d-4614-94c4-67ce114bb0e9_0.png"
image_path = room_folder / "000.png"

canvas, warped_img = prepare_masks_and_image(target_mask_path, occ_mask_path, image_path)

# can be single image or multiple images
images = [
    warped_img,
]

masks = [
    canvas,
]


# Run the pipeline
outputs = pipeline.run_multi_image(
    images,
    masks,
    seed=1,
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": 12,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 12,
        "cfg_strength": 3,
    },
    save_sparse_steps=True,
)


sparse_structure_dir = f"{scratch}/room/output/amodal3r/sparse_structure"
os.makedirs(sparse_structure_dir, exist_ok=True)

#saving the decoded sparse voxel grids at each step
sparse_steps = outputs['sparse_steps']
for step_idx, coords in enumerate(sparse_steps):
    coords_np = coords.cpu().numpy() if hasattr(coords, 'cpu') else coords
    np.savez_compressed(os.path.join(sparse_structure_dir, f"sparse_step_{step_idx:02d}.npz"), coords=coords_np)

# save as gif
video_gs = render_utils.render_video(outputs['gaussian'][0], bg_color=(1, 1, 1))['color']
video_mesh = render_utils.render_video(outputs['mesh'][0], bg_color=(1, 1, 1))['normal']
video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
imageio.mimsave(os.path.join(output_dir, "sample_multi.gif"), video, fps=30)

# save multi-view gs and mesh
gaussian = outputs['gaussian'][0]
multi_view_gs,_,_ = render_utils.render_multiview(gaussian, nviews=8, bg_color=(1, 1, 1))
multi_view_gs = multi_view_gs['color']
for i in range(8):
    output = multi_view_gs[i]
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{i:03d}_gs.png"), output)

mesh = outputs['mesh'][0]
multi_view_mesh,_,_ = render_utils.render_multiview(mesh, nviews=8, bg_color=(1, 1, 1))
multi_view_mesh = multi_view_mesh['normal']
for i in range(8):
    output = multi_view_mesh[i]
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{i:03d}_mesh.png"), output)

# # save mesh
save_mesh(mesh, os.path.join(output_dir, "mesh.ply"))

# export glb if needed
# glb_path = os.path.join(output_dir, "mesh.glb")
# extract_glb(outputs['gaussian'][0], outputs['mesh'][0], 0.5, 1024, glb_path)
