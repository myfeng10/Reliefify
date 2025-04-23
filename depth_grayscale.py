import cv2
import numpy as np
from stl import mesh
from transformers import pipeline
import trimesh
# from skimage.restoration import denoise_tv_chambolle
import open3d as o3d


def get_depth_map_transformer(image_path):
    img = cv2.imread(image_path)
    estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")
    result = estimator(images=image_path)
    depth_map = np.array(result["depth"])
    normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    return normalized_depth, img

def fill_inner_area_from_edges(edge_img):
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(edge_img)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    return filled_mask

def compute_detail_map(normalized_depth, img, depth_layer=12):
    gray_img = np.mean(img, axis=2).astype(np.uint8)
    global_edges = cv2.Canny(gray_img, 50, 150)
    partition_thresholds = np.linspace(0, 1, depth_layer + 1) ** 2
    detail_map = []
    for i in range(depth_layer):
        layer_mask = (normalized_depth >= partition_thresholds[i]) & (normalized_depth < partition_thresholds[i+1])
        layer_detail_edge = np.zeros_like(global_edges, dtype=np.float32)
        layer_detail_edge[layer_mask] = global_edges[layer_mask] / 255.0
        detail_map.append(layer_detail_edge)
    combined_detail_map = np.sum(np.array(detail_map), axis=0)
    return combined_detail_map

def create_watertight_and_smoothed_mesh(
    normalized_depth, img,
    scale, base_thickness, detail_scale, grayscale_detail_weight,
    stl_filename="model.stl",
    smooth_iters=15,
    lamb=0.5,            # positive smoothing factor
    mu=-0.53            # negative “reverse” factor to prevent shrinkage
):
    H, W = normalized_depth.shape
    detail_map = compute_detail_map(normalized_depth, img)
    gray_img = np.mean(img, axis=2).astype(np.float32) / 255.0
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 2)
    gray_detail_scaled = gray_img * grayscale_detail_weight
    filled_global = fill_inner_area_from_edges(cv2.Canny(gray_img.astype(np.uint8) * 255, 50, 150)) / 255.0
    global_edges_dilated = cv2.dilate(cv2.Canny(gray_img.astype(np.uint8) * 255, 50, 150), np.ones((3, 3), np.uint8), iterations=1) / 255.0

    #Test This More
    height_map = (
        cv2.GaussianBlur(normalized_depth, (5, 5), 1.5) * scale +
        filled_global * detail_scale +
        global_edges_dilated * 1.0 +
        gray_detail_scaled + detail_map
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

    stl_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    stl_mesh.vectors = vertices[np.array(faces)]
    stl_mesh.save(stl_filename)
    print(f"Successfully generated STL: {stl_filename}")



    # mesh_o3d = o3d.geometry.TriangleMesh(
    # o3d.utility.Vector3dVector(vertices),
    # o3d.utility.Vector3iVector(faces))

    #     # 1) Laplacian
    # mesh_o3d = mesh_o3d.filter_smooth_laplacian(number_of_iterations=10)

    # # 2) Taubin (volume‑preserving)
    # mesh_o3d = mesh_o3d.filter_smooth_taubin(number_of_iterations=10)

    # # --- ADD THESE TWO LINES ---
    # mesh_o3d.compute_triangle_normals()
    # mesh_o3d.compute_vertex_normals()

    # # export back to STL
    # o3d.io.write_triangle_mesh(stl_filename, mesh_o3d)
    
def generate_printable_model(
    image_path, scale=150, base_thickness=8,
    detail_scale=5, grayscale_detail_weight=5,
    stl_filename="output.stl",
    smooth_iters=15
):
    normalized_depth, img = get_depth_map_transformer(image_path)
    create_watertight_and_smoothed_mesh(
        normalized_depth, img, scale, base_thickness,
        detail_scale, grayscale_detail_weight,
        stl_filename,
        smooth_iters=smooth_iters
    )

# Example call
generate_printable_model(
    "image/cat/cat_ai.png",
     scale=100, base_thickness=10,
     detail_scale=10, grayscale_detail_weight=8,
     stl_filename="model/baseline/cat_ai.stl",
     smooth_iters=12
)