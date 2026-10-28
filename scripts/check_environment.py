"""Small runtime checks for this machine's compiled environment; no dataset needed."""
import importlib
import importlib.metadata
import sys

import numpy as np
import torch
import torchvision


def main():
    print('Python:', sys.version.split()[0], flush=True)
    print('PyTorch:', torch.__version__, 'torchvision:', torchvision.__version__, flush=True)
    print('CUDA:', torch.version.cuda, 'NumPy:', np.__version__, flush=True)
    assert torch.cuda.is_available(), 'CUDA is unavailable'
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    for package in ('scipy', 'open3d', 'gstaichi', 'libigl', 'pybind11', 'opencv-python'):
        print(package + ':', importlib.metadata.version(package), flush=True)
    for module in ('cv2', 'scipy', 'PIL', 'plyfile', 'trimesh', 'open3d', 'lpips',
                   'mediapy', 'matplotlib', 'tensorboard', 'gstaichi', 'igl', 'yaml',
                   'sklearn', 'skimage', 'diff_triangle_rasterization._C', 'simple_knn._C',
                   'rdel.lbvh_bind', '_nvdiffrast_c'):
        importlib.import_module(module)
    print('PASS: dependency and compiled-extension imports', flush=True)

    x = torch.randn(32, 32, device='cuda', requires_grad=True)
    (x @ x.T).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.], [2., 2., 3., 3.]], device='cuda')
    scores = torch.tensor([.9, .8, .7], device='cuda')
    assert torchvision.ops.nms(boxes, scores, .5).tolist() == [0, 2]
    print('PASS: PyTorch CUDA forward/backward and torchvision NMS', flush=True)

    from simple_knn._C import distCUDA2
    torch.manual_seed(0)
    points = torch.randn(32, 3, device='cuda')
    reference = torch.cdist(points, points).square()
    reference.fill_diagonal_(float('inf'))
    reference = reference.topk(3, largest=False).values.mean(dim=1)
    torch.testing.assert_close(distCUDA2(points), reference, rtol=1e-4, atol=1e-5)
    print('PASS: simple-knn CUDA distances against brute force', flush=True)

    import nvdiffrast.torch as dr
    context = dr.RasterizeCudaContext()
    positions = torch.tensor([[[-.8, -.8, 0., 1.], [.8, -.8, 0., 1.], [0., .8, 0., 1.]]],
                             device='cuda', requires_grad=True)
    faces_cuda = torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda')
    raster, _ = dr.rasterize(context, positions, faces_cuda, resolution=[32, 32])
    assert (raster[..., 3] > 0).any()
    raster[..., 0].sum().backward()
    assert torch.isfinite(positions.grad).all()
    print('PASS: nvdiffrast CUDA rasterization and backward', flush=True)

    import igl
    import open3d as o3d
    from rdel import lbvh_bind
    from intersect import upload_bvh_and_mesh, query_segments_any_hit
    import gstaichi as ti
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    np.testing.assert_allclose(igl.doublearea(vertices, faces), [1.])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                   o3d.utility.Vector3iVector(faces))
    np.testing.assert_allclose(mesh.get_surface_area(), .5)
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    assert ti.lang.impl.current_cfg().arch == ti.cuda, 'gstaichi fell back from CUDA'
    bvh = lbvh_bind.build_lbvh(vertices, faces)
    upload_bvh_and_mesh(bvh, vertices, faces)
    starts = torch.tensor([[.25, .25, -1.], [2., 2., -1.]], dtype=torch.float32)
    ends = torch.tensor([[.25, .25, 1.], [2., 2., 1.]], dtype=torch.float32)
    assert query_segments_any_hit(starts, ends).tolist() == [True, False]
    ti.sync()
    torch.cuda.synchronize()
    print('PASS: libigl, Open3D, effrdel BVH, and gstaichi CUDA intersection', flush=True)
    print('All environment checks passed.', flush=True)


if __name__ == '__main__':
    main()
