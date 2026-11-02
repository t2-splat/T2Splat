# Test organization maintained by zhtang2004@gmail.com.
"""Numerical regression tests for paper Eqs. 17/18, runnable without the renderer.

Run: python -m unittest -v tests.test_heartbeat_momentum
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "heartbeat_momentum_test_module", Path(__file__).resolve().parents[1] / "scene/heartbeat.py")
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)

ATTRS = {
    "vertices": "vertices",
    "f_dc": "_features_dc",
    "f_rest": "_features_rest",
    "vertex_weight": "vertex_weight",
}
MOMENTS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


class MomentumTests(unittest.TestCase):
    device = "cpu"

    def make_model(self, xyz=None, faces=None, initialized=True, amsgrad=True):
        if xyz is None:
            xyz = [[0, 0, 0], [2, 0, 0], [0, 1, 0], [-6, 0, 0]]
        if faces is None:
            # Face areas 1 and 3 -> areas at endpoints (0,1) have ratio 4:1.
            faces = [[0, 1, 2], [0, 2, 3]]
        n = len(xyz)
        tensors = [torch.tensor(xyz, dtype=torch.float32, device=self.device),
                   torch.zeros(n, 1, 3, device=self.device),
                   torch.zeros(n, 15, 3, device=self.device),
                   torch.zeros(n, 1, device=self.device)]
        model = SimpleNamespace(opacity_floor=0.0)
        groups = []
        for i, ((name, attr), data) in enumerate(zip(ATTRS.items(), tensors)):
            param = torch.nn.Parameter(data)
            setattr(model, attr, param)
            groups.append({"name": name, "params": [param], "lr": 1e-4 * (i + 1)})
        model.optimizer = torch.optim.Adam(groups, amsgrad=amsgrad, eps=1e-15)
        model._triangle_indices = torch.tensor(
            faces, dtype=torch.int32, device=self.device)
        if initialized:
            for i, group in enumerate(groups):
                p = group["params"][0]
                rows = torch.arange(n, dtype=p.dtype, device=p.device)
                shape = (n,) + (1,) * (p.ndim - 1)
                m = (rows * 2 + i + 1).reshape(shape).expand_as(p).clone()
                model.optimizer.state[p] = {
                    "step": torch.tensor(7.0),
                    "exp_avg": m,
                    "exp_avg_sq": m.square() + 1,
                    "max_exp_avg_sq": m.square() + 3,
                }
        return model

    def arrays(self, model):
        arrays = [getattr(model, attr).detach().cpu().numpy().copy()
                  for attr in ATTRS.values()]
        arrays.append(model._triangle_indices.cpu().numpy().astype(np.int64))
        return arrays

    def snapshot(self, model):
        return {
            group["name"]: {key: value.clone()
                            for key, value in model.optimizer.state[group["params"][0]].items()}
            for group in model.optimizer.param_groups
        }

    def collapse(self, model):
        return hb.batch_edge_collapse(
            *self.arrays(model), edges=np.array([[0, 1]], dtype=np.int64),
            opacity_floor=0.0, collapse_ratio=0.25, use_qem=False)

    def stitch(self, model, data, keep, edges, weights, parents, std=0.0):
        hb.surgery_and_stitch_params(
            model, model.optimizer, keep, len(parents), *data,
            collapse_edges=edges, collapse_weights=weights,
            split_parents=parents, split_momentum_std=std)

    def assert_can_step(self, model, previous_step):
        model.optimizer.zero_grad(set_to_none=True)
        loss = sum((getattr(model, attr).square() + getattr(model, attr)).mean()
                   for attr in ATTRS.values())
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        for group in model.optimizer.param_groups:
            p = group["params"][0]
            self.assertTrue(torch.isfinite(p).all())
            state = model.optimizer.state[p]
            self.assertEqual(state["step"].item(), previous_step + 1)
            for key in MOMENTS:
                self.assertEqual(state[key].shape, p.shape)
                self.assertTrue(torch.isfinite(state[key]).all())
            self.assertTrue((state["exp_avg_sq"] >= 0).all())
            self.assertTrue((state["max_exp_avg_sq"] >= state["exp_avg_sq"]).all())

    def test_collapse_uses_pre_edit_area_for_all_moments(self):
        model = self.make_model()
        before = self.snapshot(model)
        old_params = [g["params"][0] for g in model.optimizer.param_groups]
        lrs = [g["lr"] for g in model.optimizer.param_groups]
        *data, keep, edges, weights = self.collapse(model)
        np.testing.assert_array_equal(edges, [[0, 1]])
        np.testing.assert_allclose(weights, [[0.8, 0.2]])
        self.stitch(model, data, keep, edges, weights, np.empty(0, dtype=np.int64))
        for group, old in zip(model.optimizer.param_groups, old_params):
            state = model.optimizer.state[group["params"][0]]
            original = before[group["name"]]
            for key in MOMENTS:
                expected = original[key][[0, 2, 3]].clone()
                expected[0] = 0.8 * original[key][0] + 0.2 * original[key][1]
                torch.testing.assert_close(state[key], expected)
            self.assertEqual(state["step"].item(), 7)
            self.assertNotIn(old, model.optimizer.state)
        self.assertEqual(lrs, [g["lr"] for g in model.optimizer.param_groups])
        self.assert_can_step(model, 7)

    def test_zero_area_collapse_falls_back_to_equal_weights(self):
        model = self.make_model(xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                                faces=[[0, 1, 2]])
        before = self.snapshot(model)
        *data, keep, edges, weights = self.collapse(model)
        np.testing.assert_allclose(weights, [[0.5, 0.5]])
        self.stitch(model, data, keep, edges, weights, np.empty(0, dtype=np.int64))
        for group in model.optimizer.param_groups:
            state = model.optimizer.state[group["params"][0]]
            for key in MOMENTS:
                expected = before[group["name"]][key][:2].mean(dim=0)
                torch.testing.assert_close(state[key][0], expected)
        self.assert_can_step(model, 7)

    def split_model(self, model, importance):
        return hb.batch_vertex_split(*self.arrays(model),
                                     importance_np=np.asarray(importance),
                                     top_k_percent=67.0)

    def test_split_tracks_requester_and_resolves_duplicate_edge(self):
        for importance, expected_parent in [([0., 5., 9.], 2),
                                            ([0., 9., 9.], 1)]:
            with self.subTest(importance=importance):
                model = self.make_model(
                    xyz=[[0, 0, 0], [1, 0, 0], [0, 2, 0]], faces=[[0, 1, 2]])
                *data, count, parents = self.split_model(model, importance)
                self.assertEqual(count, 1)
                np.testing.assert_array_equal(parents, [expected_parent])
                np.testing.assert_allclose(data[0][-1], [0.5, 1.0, 0.0])

    def test_split_inherits_moments_with_opposite_first_moment_noise(self):
        model = self.make_model(
            xyz=[[0, 0, 0], [1, 0, 0], [0, 2, 0]], faces=[[0, 1, 2]])
        before = self.snapshot(model)
        *data, count, parents = self.split_model(model, [0., 5., 9.])
        self.assertEqual(count, 1)
        torch.manual_seed(43)
        self.stitch(model, data, np.ones(3, dtype=bool),
                    np.empty((0, 2), dtype=np.int64), np.empty((0, 2)),
                    parents, std=1e-3)
        for group in model.optimizer.param_groups:
            original = before[group["name"]]
            state = model.optimizer.state[group["params"][0]]
            # Parent 2 and child 3 retain their mean; unrelated endpoint 1
            # must not be used as the source or receive perturbations.
            torch.testing.assert_close(
                (state["exp_avg"][2] + state["exp_avg"][3]) / 2,
                original["exp_avg"][2])
            self.assertFalse(torch.equal(state["exp_avg"][2], original["exp_avg"][2]))
            for key in MOMENTS:
                torch.testing.assert_close(state[key][:2], original[key][:2],
                                           rtol=0, atol=0)
            for key in MOMENTS[1:]:
                torch.testing.assert_close(state[key][2], original[key][2],
                                           rtol=0, atol=0)
                torch.testing.assert_close(state[key][3], original[key][2],
                                           rtol=0, atol=0)
        self.assert_can_step(model, 7)

    def test_split_noise_cannot_explode_with_unobserved_or_tiny_gradients(self):
        # Match the training optimizer's eps and a late DTE step. Vertices
        # unseen by the next view receive zero gradients, including fresh SH
        # coefficients with little or no second-moment history.
        for amsgrad in (False, True):
            for variance in (0.0, 1e-30, 1e-24, 1e-16, 1e-12):
                with self.subTest(amsgrad=amsgrad, variance=variance):
                    model = self.make_model(
                        xyz=[[0, 0, 0], [1, 0, 0], [0, 2, 0]],
                        faces=[[0, 1, 2]], amsgrad=amsgrad)
                    for group in model.optimizer.param_groups:
                        state = model.optimizer.state[group["params"][0]]
                        state["step"].fill_(13000)
                        state["exp_avg"].zero_()
                        state["exp_avg_sq"].fill_(variance)
                        if amsgrad:
                            state["max_exp_avg_sq"].fill_(variance)
                        else:
                            del state["max_exp_avg_sq"]
                    *data, _, parents = self.split_model(model, [0., 5., 9.])
                    torch.manual_seed(43)
                    self.stitch(model, data, np.ones(3, dtype=bool),
                                np.empty((0, 2), dtype=np.int64), np.empty((0, 2)),
                                parents, std=1e-3)
                    before = {}
                    for group in model.optimizer.param_groups:
                        p = group["params"][0]
                        before[group["name"]] = p.detach().clone()
                        p.grad = torch.zeros_like(p)
                    model.optimizer.step()
                    for group in model.optimizer.param_groups:
                        p = group["params"][0]
                        update = (p.detach() - before[group["name"]]).abs().max().item()
                        self.assertTrue(torch.isfinite(p).all())
                        # A dimensionless std of 1e-3 must give a small
                        # fraction of the learning rate, even when v -> 0.
                        self.assertLess(update, 0.01 * group["lr"])
                        if variance == 0:
                            self.assertEqual(update, 0.0)

    def test_combined_edit_inherits_merged_post_compaction_parent(self):
        model = self.make_model()
        before = self.snapshot(model)
        *collapsed, keep, edges, weights = self.collapse(model)
        *data, count, parents = hb.batch_vertex_split(
            *collapsed, importance_np=np.array([10., 0., 0.]), top_k_percent=1.0)
        np.testing.assert_array_equal(parents, [0])
        self.assertEqual(count, 1)
        self.stitch(model, data, keep, edges, weights, parents)
        for group in model.optimizer.param_groups:
            original = before[group["name"]]
            state = model.optimizer.state[group["params"][0]]
            for key in MOMENTS:
                merged = 0.8 * original[key][0] + 0.2 * original[key][1]
                torch.testing.assert_close(state[key][0], merged)
                torch.testing.assert_close(state[key][-1], merged)
                torch.testing.assert_close(state[key][1:3], original[key][2:4],
                                           rtol=0, atol=0)
        self.assert_can_step(model, 7)

    def test_noop_preserves_history(self):
        model = self.make_model()
        before = self.snapshot(model)
        hb.heartbeat_step(model, collapse_ratio=0, split_top_k=0)
        after = self.snapshot(model)
        for name in before:
            for key in before[name]:
                torch.testing.assert_close(before[name][key], after[name][key],
                                           rtol=0, atol=0)

    def test_lazy_adam_state_is_initialized_by_next_step(self):
        model = self.make_model(initialized=False)
        hb.heartbeat_step(model, mode="split", split_top_k=25)
        self.assertEqual(len(model.optimizer.state), 0)
        self.assert_can_step(model, 0)

    def test_real_heartbeat_modes_can_continue_training(self):
        xyz = [[x, y, 0] for y in range(3) for x in range(3)]
        faces = []
        for y in range(2):
            for x in range(2):
                a = y * 3 + x
                faces.extend([[a, a + 1, a + 3], [a + 1, a + 4, a + 3]])
        for mode in ("collapse", "split", "both"):
            with self.subTest(mode=mode):
                model = self.make_model(xyz=xyz, faces=faces)
                # Training consumes gradients first, then edits topology.
                for previous_step in (7, 8):
                    self.assert_can_step(model, previous_step)
                    hb.init_heartbeat_stats(model)
                    model._hb_grad_accum.copy_(
                        torch.arange(len(model.vertices), device=self.device))
                    model._hb_obs_count.fill_(1)
                    hb.heartbeat_step(model, mode=mode, collapse_ratio=0.1,
                                      split_top_k=20, enable_safety=True)
                    self.assertTrue((model._triangle_indices >= 0).all())
                    self.assertTrue(
                        (model._triangle_indices < len(model.vertices)).all())
                self.assert_can_step(model, 9)

    def test_invalid_noise_rejected_before_mutation(self):
        model = self.make_model()
        vertices = model.vertices
        for std in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                hb.heartbeat_step(model, split_momentum_std=std)
            self.assertIs(model.vertices, vertices)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class CudaMomentumTests(MomentumTests):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
