import cv2
import torch
import urllib.request
import ssl
import numpy as np
from stl import mesh
import open3d as o3d
import matplotlib.pyplot as plt
from transformers import pipeline


def get_depth_map_transformer(image_path):
    img = cv2.imread(image_path)
    estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")
    result = estimator(images=image_path)
    depth_map = np.array(result["depth"])
    normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    return normalized_depth, img


def create_watertight_and_smoothed_mesh(
    normalized_depth, img,
    scale, base_thickness, grayscale_detail_weight,
    stl_filename="model.stl",
    smooth_type="none",
    smooth_iters=12

):
    H, W = normalized_depth.shape

    # adding detail by grayscale
    gray_img = np.mean(img, axis=2).astype(np.float32) / 255.0
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 2)
    gray_detail_scaled = gray_img * grayscale_detail_weight

    # adding detail by edge detection  
    global_edges_dilated = cv2.dilate(cv2.Canny(gray_img.astype(np.uint8) * 255, 50, 150), np.ones((3, 3), np.uint8), iterations=1) / 255.0


    height_map = (
        normalized_depth * scale +
        global_edges_dilated +
        gray_detail_scaled 
    )

    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    z_top = height_map.astype(int)
    z_bottom = np.full_like(z_top, -base_thickness)
    vertices_top = np.column_stack((xx.ravel(), yy.ravel(), z_top.ravel()))
    vertices_bottom = np.column_stack((xx.ravel(), yy.ravel(), z_bottom.ravel()))
    vertices = np.vstack((vertices_top, vertices_bottom))
    total_vertices = H * W

    faces = []
    for i in range(H - 1):
        for j in range(W - 1):
            v1 = i * W + j
            v2 = i * W + (j + 1)
            v3 = (i + 1) * W + j
            v4 = (i + 1) * W + (j + 1)
            faces.append([v1, v2, v3])
            faces.append([v3, v2, v4])
    for i in range(H - 1):
        for j in range(W - 1):
            v1 = i * W + j + total_vertices
            v2 = i * W + (j + 1) + total_vertices
            v3 = (i + 1) * W + j + total_vertices
            v4 = (i + 1) * W + (j + 1) + total_vertices
            faces.append([v1, v3, v2])
            faces.append([v3, v4, v2])
    for i in range(H - 1):
        v_top1 = i * W
        v_top2 = (i + 1) * W
        v_bot1 = v_top1 + total_vertices
        v_bot2 = v_top2 + total_vertices
        faces.append([v_top1, v_top2, v_bot1])
        faces.append([v_top2, v_bot2, v_bot1])
    for i in range(H - 1):
        v_top1 = i * W + (W - 1)
        v_top2 = (i + 1) * W + (W - 1)
        v_bot1 = v_top1 + total_vertices
        v_bot2 = v_top2 + total_vertices
        faces.append([v_top1, v_bot1, v_top2])
        faces.append([v_top2, v_bot1, v_bot2])
    for j in range(W - 1):
        v_top1 = j
        v_top2 = j + 1
        v_bot1 = v_top1 + total_vertices
        v_bot2 = v_top2 + total_vertices
        faces.append([v_top1, v_bot1, v_top2])
        faces.append([v_top2, v_bot1, v_bot2])
    for j in range(W - 1):
        v_top1 = (H - 1) * W + j
        v_top2 = (H - 1) * W + j + 1
        v_bot1 = v_top1 + total_vertices
        v_bot2 = v_top2 + total_vertices
        faces.append([v_top1, v_top2, v_bot1])
        faces.append([v_top2, v_bot2, v_bot1])

    if smooth_type == "none":
        stl_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        stl_mesh.vectors = vertices[np.array(faces)]
        stl_mesh.save(stl_filename)
        print(f"Successfully generated STL: {stl_filename}")
    elif smooth_type == "smooth":

        mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces))


        mesh_o3d = mesh_o3d.filter_smooth_taubin(number_of_iterations=smooth_iters)

        mesh_o3d.compute_triangle_normals()
        mesh_o3d.compute_vertex_normals()

        # export back to STL
        o3d.io.write_triangle_mesh(stl_filename, mesh_o3d)


def generate_printable_model(
    image_path, smooth_iters, scale=150, base_thickness=8,
    detail_scale=5, grayscale_detail_weight=5,
    stl_filename="output.stl",
    smooth_type="default",
):
    normalized_depth, img = get_depth_map_transformer(image_path)
    create_watertight_and_smoothed_mesh(
        normalized_depth, img, scale, base_thickness,
        detail_scale, grayscale_detail_weight,
        stl_filename,
        smooth_type=smooth_type,
        smooth_iters=smooth_iters
    )


smooth_iters = 100
generate_printable_model(
    "image/baseline/girl_figure.png",
     scale=100, base_thickness=10,
     detail_scale=10, grayscale_detail_weight=8,
     smooth_iters=smooth_iters,
     stl_filename="model/dev/girl_figure"+str(smooth_iters)+"1.stl",
)