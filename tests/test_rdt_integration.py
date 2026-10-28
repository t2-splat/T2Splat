# Test organization maintained by zhtang2004@gmail.com.
"""CUDA interface tests: RDT preserves live Adam state and winding is render-safe."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA renderer required")
class RDTIntegration(unittest.TestCase):
    def make_model(self):
        from tests.test_heartbeat_momentum import MomentumTests
        helper = MomentumTests(); helper.device = 'cuda'
        model = helper.make_model(xyz=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
                                  faces=[[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
        return model

    def test_live_parameters_and_moments_preserved(self):
        from scene.triangle_model import TriangleModel
        model = self.make_model()
        params = [g['params'][0] for g in model.optimizer.param_groups]
        values = [p.detach().clone() for p in params]
        moments = [{k: v.clone() for k, v in model.optimizer.state[p].items()} for p in params]
        original_faces = model._triangle_indices
        def fake_rdt(vertices, faces, **kwargs):
            selected, stats = kwargs['face_selector'](vertices, faces, faces)
            return selected, {'selector_stats':stats}
        with patch('scene.triangle_model.rdel.run', side_effect=fake_rdt):
            TriangleModel.run_restricted_delaunay(model, return_debug=True, backend='surface')
        self.assertIsNot(model._triangle_indices, original_faces)
        for p, expected, state in zip(params, values, moments):
            self.assertTrue(any(p is g['params'][0] for g in model.optimizer.param_groups))
            torch.testing.assert_close(p, expected, rtol=0, atol=0)
            for k, v in state.items(): torch.testing.assert_close(model.optimizer.state[p][k], v, rtol=0, atol=0)
        self.assertEqual(len(model.image_size), 4)
        sum(p.square().mean() for p in params).backward()
        model.optimizer.step()
        self.assertTrue(all(torch.isfinite(p).all() for p in params))
        self.assertTrue(all(model.optimizer.state[p]['step'].item() == 8 for p in params))

    def test_failure_keeps_original_faces(self):
        from scene.triangle_model import TriangleModel
        model = self.make_model(); original = model._triangle_indices
        with patch('scene.triangle_model.rdel.run', side_effect=RuntimeError('probe error')):
            with self.assertRaises(RuntimeError): TriangleModel.run_restricted_delaunay(model)
        self.assertIs(model._triangle_indices, original)

    def test_invalid_backend_and_texture_guard(self):
        from scene.triangle_model import TriangleModel
        model = self.make_model()
        with self.assertRaises(ValueError): TriangleModel.run_restricted_delaunay(model, backend='bad')
        model.face_textures = torch.zeros(4,8,8,3,device='cuda')
        with self.assertRaises(RuntimeError): TriangleModel.run_restricted_delaunay(model, backend='surface')

    def test_winding_image_and_gradient_invariance(self):
        import diff_triangle_rasterization as raster
        from utils.graphics_utils import getProjectionMatrix
        settings = raster.TriangleRasterizationSettings(image_height=32,image_width=40,tanfovx=1.,tanfovy=1.,
            bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=torch.eye(4,device='cuda'),
            projmatrix=getProjectionMatrix(.01,100.,torch.pi/2,torch.pi/2).T.cuda(),
            sh_degree=0,campos=torch.zeros(3,device='cuda'),prefiltered=False,debug=False)
        def run(reverse):
            vertices = torch.tensor([[-.8,-.7,2.],[.9,-.6,2.1],[-.1,.9,2.2]],device='cuda',requires_grad=True)
            weights = torch.tensor([.6,.8,.7],device='cuda',requires_grad=True)
            colors = torch.tensor([[.2,.4,.6],[.9,.4,.2],[.1,.6,.8]],device='cuda',requires_grad=True)
            f = [[0,2,1]] if reverse else [[0,1,2]]
            out = raster.TriangleRasterizer(settings)(vertices=vertices,
                triangles_indices=torch.tensor(f,dtype=torch.int32,device='cuda'),vertex_weights=weights,
                sigma=.5,scaling=torch.zeros(1,device='cuda'),colors_precomp=colors)
            # Channel 6 contains discrete face IDs, undefined on background in
            # the legacy renderer. It is not a continuous training output.
            (out[0].square().mean()+.001*out[3][:6].mean()).backward()
            return [out[0],out[3][:6],vertices.grad,weights.grad,colors.grad]
        for a,b in zip(run(False),run(True)):
            torch.testing.assert_close(a,b,rtol=1e-4,atol=2e-6)

if __name__=='__main__': unittest.main()
