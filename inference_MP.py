import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import imageio
from PIL import Image
from amodal3r.pipelines import Amodal3RImageTo3DPipeline
from amodal3r.utils import render_utils, postprocessing_utils
import cv2
import trimesh
from pathlib import Path
import torch

import multiprocessing as multip

scratch = os.environ["SCRATCH"]


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



segments_folder = Path(f'{scratch}/new_WOD_object_masks')
segment_names = sorted(os.listdir(segments_folder))



def collect_objects(folder):
    objects = []
    for p in Path(folder).glob("*_object.png"):
        name = p.name
        objects.append(name[: -len("_object.png")])
    return objects


def worker_build_3D_object(worker_id: int, world_size: int, cuda_id: str):

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    try:
        import torch
        if torch.cuda.device_count() > 0:
            torch.cuda.set_device(0)
            torch.set_num_threads(1)  # avoid CPU thread thrash
    except Exception:
        pass

    pipeline = Amodal3RImageTo3DPipeline.from_pretrained("./Amodal3R")
    pipeline.cuda()


    #Segment to process
    def is_segment_done(key: str) -> bool:
        outdir = segments_folder / key
        if not outdir.exists():
            return False
        # Any composed RGBA beyond the no_obj image counts as 'done'
        #checking if there is a completed.txt
        done_file = segments_folder / key / "completed.txt"
        if done_file.exists():
            return True
        return False

    all_keys = segment_names

    # Only keep segments that are NOT finished yet
    pending_keys = [k for k in all_keys if not is_segment_done(k)]

    # Shard ONLY the pending list
    keys = [k for idx, k in enumerate(pending_keys) if idx % world_size == worker_id]

    print(f"[worker {worker_id+1}/{world_size}] pending={len(pending_keys)} / total={len(all_keys)}")
    print(f"[worker {worker_id+1}/{world_size}] assigned {len(keys)} segments after filtering")
    if not keys:
        print(f"[worker {worker_id+1}/{world_size}] nothing to do.")
        return


    for segment in keys:
        print(f"Processing segment: {segment}")

        #gathering the objects
        segment_folder = segments_folder / segment
        object_list = collect_objects(segment_folder)


        for object_name in object_list:
            print(f"Processing object: {object_name}")

            output_dir = f"{scratch}/new_WOD_output_objects/{segment}/{object_name}/"
            os.makedirs(output_dir, exist_ok=True)


            if os.path.exists(os.path.join(output_dir, "gaussian.ply")):
                print(f"Skipping {object_name} in {segment}, already processed.")
                continue


            # can be single image or multiple images
            images = [
                Image.open(f"{scratch}/new_WOD_object_masks/{segment}/{object_name}_object.png"),
            ]

            masks = [
                Image.open(f"{scratch}/new_WOD_object_masks/{segment}/{object_name}_mask.png").convert("L"),
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
            )

            # save as gif
            video_gs = render_utils.render_video(outputs['gaussian'][0], bg_color=(1, 1, 1))['color']
            video_mesh = render_utils.render_video(outputs['mesh'][0], bg_color=(1, 1, 1))['normal']
            video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
            imageio.mimsave(os.path.join(output_dir, "sample_multi.gif"), video, fps=30)

            # save multi-view gs and mesh
            gaussian = outputs['gaussian'][0]
            gaussian.save_ply(os.path.join(output_dir, "gaussian.ply"))
            
            # mesh = outputs['mesh'][0]

            # # save mesh
            # save_mesh(mesh, os.path.join(output_dir, "mesh.ply"))

            # export glb if needed
            # glb_path = os.path.join(output_dir, "mesh.glb")
            # extract_glb(outputs['gaussian'][0], outputs['mesh'][0], 0.5, 1024, glb_path)
        
        # Mark segment as completed
        done_file = segments_folder / segment / "completed.txt"
        with open(done_file, "w") as f:
            f.write("done\n")
        




def _gpu_id_list():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip() != "":
        # Respect the mask order; map to integers if possible
        ids = [s.strip() for s in env.split(",") if s.strip() != ""]
        return ids
    # Fall back to local ordinal IDs
    return [str(i) for i in range(torch.cuda.device_count())]


def main():
    gpu_ids = _gpu_id_list()
    if not gpu_ids:
        raise RuntimeError("No GPUs visible (CUDA_VISIBLE_DEVICES is empty) — refusing to run.")

    world_size = len(gpu_ids)
    rank_to_cuda = gpu_ids
    print(f"[launcher] multi-GPU mode with {world_size} workers: {rank_to_cuda}", flush=True)



    ctx = multip.get_context("spawn")
    with ctx.Pool(world_size) as pool:
        pool.starmap(
            worker_build_3D_object,
            [(rank, world_size, rank_to_cuda[rank]) for rank in range(world_size)]
    )



if __name__ == "__main__":
    try:
        multip.set_start_method("spawn")
    except RuntimeError:
        pass
    main()

