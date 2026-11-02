# Test organization maintained by zhtang2004@gmail.com.
"""Progressive visibility collapse: topology, visibility, Adam and CUDA integration."""
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
import ast
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import numpy as np
import torch

from tests import test_heartbeat_momentum as momentum
hb, MOMENTS = momentum.hb, momentum.MOMENTS
from tests.test_training_topology import grid, TETRA_X, TETRA_F
from arguments import OptimizationParams
from utils.mesh_topology import mesh_topology_metrics
from utils.visibility_pruning import (VisibilityPruningSchedule, collect_face_visibility,
                                      finalize_visibility_pruning, require_surface_topology)


class ScheduleTests(unittest.TestCase):
    def test_exact_events_ramp_and_recovery(self):
        schedule = VisibilityPruningSchedule(mode="progressive_collapse")
        events = [(i, schedule.threshold_at(i)) for i in range(30001)
                  if schedule.threshold_at(i) is not None]
        self.assertEqual([i for i, _ in events], list(range(21000, 27001, 1000)))
        np.testing.assert_allclose([v for _, v in events], np.linspace(.05, .5, 7))
        self.assertIsNone(schedule.threshold_at(30000))
        self.assertIsNone(VisibilityPruningSchedule().threshold_at(21000))

    def test_none_has_no_events_or_progressive_schedule_requirements(self):
        schedule = VisibilityPruningSchedule(mode="none", start=0, end=0, interval=0)
        self.assertFalse(schedule.enabled)
        self.assertTrue(all(schedule.threshold_at(i) is None for i in range(35001)))
        schedule.validate_training(SimpleNamespace(), 11000, 28000)

    def test_invalid_schedule_and_training_conflicts_rejected(self):
        schedule = VisibilityPruningSchedule(mode="progressive_collapse")
        for kwargs in (dict(mode="invalid"), dict(interval=0), dict(end=26001), dict(end=21000),
                       dict(threshold_start=float('nan')), dict(threshold_end=1.1),
                       dict(max_face_ratio=0), dict(max_face_ratio=float('nan'))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(schedule, **kwargs)
        valid = dict(rdt_backend="surface", iterations=30000, enable_heartbeat=True,
                     heartbeat_end_iter=20000, heartbeat_safety=True)
        schedule.validate_training(SimpleNamespace(**valid), 11000, 28000)
        for kwargs, rdt, texture in ((dict(iterations=27000), 11000, None),
                (dict(rdt_backend="legacy"), 11000, None),
                (dict(heartbeat_end_iter=22000), 11000, None),
                (dict(heartbeat_safety=False), 11000, None),
                ({}, 21000, None), ({}, 11000, 25000)):
            with self.subTest(kwargs=kwargs, rdt=rdt, texture=texture), self.assertRaises(ValueError):
                schedule.validate_training(SimpleNamespace(**(valid | kwargs)), rdt, texture)


class VisibilityTests(unittest.TestCase):
    def test_fresh_all_view_maximum_without_grad_or_stale_statistics(self):
        model = SimpleNamespace(vertices=torch.zeros(3, 3), _triangle_indices=torch.zeros(3, 3),
                                importance_score=torch.ones(3))
        observations = [torch.tensor([.01, .8, .1]), torch.tensor([.9, .02, .3])]
        def renderer(cam, *args):
            self.assertFalse(torch.is_grad_enabled())
            return {"max_blending": observations[cam]}
        scores = collect_face_visibility(model, [0, 1], renderer, None, None)
        torch.testing.assert_close(scores, torch.tensor([.9, .8, .3]))
        observations[0].fill_(.01); observations[1].fill_(.02)
        scores = collect_face_visibility(model, [0, 1], renderer, None, None)
        torch.testing.assert_close(scores, torch.full((3,), .02))
        torch.testing.assert_close(model.importance_score, torch.ones(3))

    def test_empty_or_corrupt_measurements_rejected(self):
        model = SimpleNamespace(vertices=torch.zeros(3, 3), _triangle_indices=torch.zeros(1, 3))
        with self.assertRaises(ValueError):
            collect_face_visibility(model, [], None, None, None)
        for value in (torch.zeros(2), torch.tensor([float('nan')])):
            with self.assertRaises(RuntimeError):
                collect_face_visibility(model, [0], lambda *args: {"max_blending": value}, None, None)

    def test_final_progressive_path_never_renders_or_prunes_faces(self):
        for mode in ("progressive_collapse", "none"):
            with self.subTest(mode=mode):
                model = SimpleNamespace(vertices=torch.zeros(3, 3), _triangle_indices=torch.tensor([[0, 1, 2]]),
                                        prune_triangles=Mock(side_effect=AssertionError("face prune")),
                                        _prune_vertices=Mock(side_effect=AssertionError("unneeded compaction")))
                render = Mock(side_effect=AssertionError("final render"))
                faces = model._triangle_indices
                stats = finalize_visibility_pruning(model, [], render, None, None, mode)
                self.assertEqual(stats["faces_after"], 1)
                self.assertIs(model._triangle_indices, faces)
                render.assert_not_called(); model.prune_triangles.assert_not_called()


    def test_legacy_final_prune_still_uses_point_five_threshold(self):
        model = SimpleNamespace(vertices=torch.zeros(3, 3), _triangle_indices=torch.tensor([[0,1,2]] * 3))
        masks = []
        def prune(mask):
            masks.append(mask.clone()); model._triangle_indices = model._triangle_indices[mask]
        model.prune_triangles = prune
        stats = finalize_visibility_pruning(model, [0],
            lambda *args: {"max_blending": torch.tensor([.49,.5,.51])}, None, None, "legacy")
        self.assertEqual(stats["faces_after"], 1)
        torch.testing.assert_close(masks[0], torch.tensor([False, False, True]))


    def test_legacy_custom_threshold_is_strict(self):
        model = SimpleNamespace(vertices=torch.zeros(3, 3),
                                _triangle_indices=torch.tensor([[0, 1, 2]] * 3))
        masks = []
        def prune(mask):
            masks.append(mask.clone())
            model._triangle_indices = model._triangle_indices[mask]
        model.prune_triangles = prune
        stats = finalize_visibility_pruning(model, [0],
            lambda *args: {"max_blending": torch.tensor([.59, .6, .61])},
            None, None, "legacy", final_visibility_threshold=.6)
        self.assertEqual(stats["faces_after"], 1)
        torch.testing.assert_close(masks[0], torch.tensor([False, False, True]))

    def test_invalid_final_threshold_rejects_before_mutation(self):
        for threshold in (-.01, 1.01, float('nan'), float('inf'), -float('inf')):
            with self.subTest(threshold=threshold):
                model = Mock()
                render = Mock()
                with self.assertRaisesRegex(ValueError, 'final_visibility_threshold'):
                    finalize_visibility_pruning(model, [0], render, None, None,
                                                'legacy', final_visibility_threshold=threshold)
                render.assert_not_called()
                model.prune_triangles.assert_not_called()
                model._prune_vertices.assert_not_called()

    def test_nonlegacy_finalization_ignores_valid_threshold_for_face_deletion(self):
        for mode in ('progressive_collapse', 'none'):
            for threshold in (0., .6, 1.):
                with self.subTest(mode=mode, threshold=threshold):
                    model = SimpleNamespace(vertices=torch.zeros(3, 3),
                        _triangle_indices=torch.tensor([[0, 1, 2]]),
                        prune_triangles=Mock(side_effect=AssertionError('face pruning')),
                        _prune_vertices=Mock(side_effect=AssertionError('unneeded compaction')))
                    render = Mock(side_effect=AssertionError('visibility render'))
                    original = model._triangle_indices
                    finalize_visibility_pruning(model, [], render, None, None,
                                                mode, final_visibility_threshold=threshold)
                    self.assertIs(model._triangle_indices, original)
                    model.prune_triangles.assert_not_called()
                    render.assert_not_called()


class TrainingRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(__file__).resolve().parents[1] / 'train.py'
        cls.tree = ast.parse(cls.source.read_text())

    def test_cli_defaults_and_overrides(self):
        parser = ArgumentParser()
        options = OptimizationParams(parser)
        default = options.extract(parser.parse_args([]))
        self.assertEqual(default.final_visibility_threshold, .5)
        self.assertFalse(default.final_only_monitoring)
        custom = options.extract(parser.parse_args([
            '--final_visibility_threshold', '.6', '--final_only_monitoring']))
        self.assertEqual(custom.final_visibility_threshold, .6)
        self.assertTrue(custom.final_only_monitoring)

    def test_actual_training_final_call_uses_configured_threshold(self):
        calls = [node for node in ast.walk(self.tree)
                 if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == 'finalize_visibility_pruning']
        self.assertEqual(len(calls), 1)
        model = SimpleNamespace(vertices=torch.zeros(3, 3),
                                _triangle_indices=torch.tensor([[0, 1, 2]] * 3))
        masks = []
        def prune(mask):
            masks.append(mask.clone())
            model._triangle_indices = model._triangle_indices[mask]
        model.prune_triangles = prune
        context = dict(triangles=model, scene=SimpleNamespace(getTrainCameras=lambda: [0]),
            render=lambda *args: {'max_blending': torch.tensor([.55, .6, .61])},
            pipe=None, background=None, visibility_schedule=VisibilityPruningSchedule(),
            opt=SimpleNamespace(final_visibility_threshold=.6),
            finalize_visibility_pruning=finalize_visibility_pruning)
        exec(compile(ast.Module(body=calls, type_ignores=[]), str(self.source), 'exec'), context)
        self.assertEqual(context['final_stats']['faces_after'], 1)
        torch.testing.assert_close(masks[0], torch.tensor([False, False, True]))

    def test_actual_training_report_receives_monitoring_flag(self):
        calls = [node for node in ast.walk(self.tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == 'training_report']
        self.assertEqual(len(calls), 1)
        report = Mock()
        context = dict(training_report=report, tb_writer=None, scene_name='test',
            iteration=1000, pixel_loss=None, loss=None, l1_loss=None,
            iter_start=SimpleNamespace(elapsed_time=lambda end: 0), iter_end=None,
            testing_iterations=[20000], scene=None, render=None, pipe=None, background=None,
            opt=SimpleNamespace(final_only_monitoring=True))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=calls[0])], type_ignores=[])),
                     str(self.source), 'exec'), context)
        self.assertTrue(report.call_args.kwargs['final_only_monitoring'])

    def test_monitoring_preserves_scalar_logs_and_default_evaluation_schedule(self):
        functions = [node for node in self.tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == 'training_report']
        self.assertEqual(len(functions), 1)
        class EvaluationReached(Exception):
            pass
        torch_stub = SimpleNamespace(cuda=SimpleNamespace(
            empty_cache=Mock(side_effect=EvaluationReached)))
        context = dict(Scene=SimpleNamespace, torch=torch_stub)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(self.source), 'exec'), context)
        for flag, iteration, evaluate in ((False, 1000, True), (False, 999, False),
                                         (True, 1000, False), (True, 20000, True)):
            with self.subTest(final_only=flag, iteration=iteration):
                writer = Mock()
                scalar = SimpleNamespace(item=lambda: .1)
                args = (writer, 'test', iteration, scalar, scalar, None, 1., [20000], None, None, None)
                kwargs = dict(final_only_monitoring=flag) if flag else {}
                if evaluate:
                    with self.assertRaises(EvaluationReached):
                        context['training_report'](*args, **kwargs)
                else:
                    context['training_report'](*args, **kwargs)
                self.assertEqual(writer.add_scalar.call_count, 3)

    def test_monitoring_skips_only_periodic_topology_branch(self):
        guards = [node for node in ast.walk(self.tree) if isinstance(node, ast.If)
                  and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                          and child.func.id == 'record_topology' and child.args
                          and isinstance(child.args[0], ast.Constant)
                          and child.args[0].value == 'periodic'
                          for statement in node.body for child in ast.walk(statement))]
        # The training function is not itself an If; only the direct periodic guard matches.
        self.assertEqual(len(guards), 1)
        code = compile(ast.Module(body=guards, type_ignores=[]), str(self.source), 'exec')
        for flag in (False, True):
            record = Mock()
            context = dict(topology_monitor=object(), topology_interval=1000, iteration=12000,
                           run_restricted_delaunay=11000, opt=SimpleNamespace(final_only_monitoring=flag),
                           record_topology=record)
            exec(code, context)
            self.assertEqual(record.call_count, 0 if flag else 1)


class CollapseTests(unittest.TestCase):
    def make(self, n=10):
        helper = momentum.MomentumTests(); xyz, faces = grid(n)
        return helper, helper.make_model(xyz=xyz, faces=faces)

    def test_multiple_events_preserve_disk_and_budget(self):
        helper, model = self.make()
        total = 0
        for event in range(4):
            before = len(model._triangle_indices)
            stats = hb.visibility_collapse_step(model, torch.zeros(before), .5, max_face_ratio=.08)
            self.assertGreater(stats["collapsed_edges"], 0)
            self.assertLessEqual(before - len(model._triangle_indices), int(before * .08))
            metrics = mesh_topology_metrics(model.vertices.detach().numpy(), model._triangle_indices.numpy())
            require_surface_topology(metrics)
            self.assertEqual(metrics["face_components"], 1)
            self.assertEqual(metrics["vertices"] - metrics["edges"] + metrics["faces"], 1)
            helper.assert_can_step(model, 7 + event)
            total += stats["removed_faces"]
        self.assertGreater(total, 0)

    def test_high_visibility_faces_survive_and_adam_history_transfers(self):
        helper, model = self.make()
        old_faces = model._triangle_indices.numpy().copy()
        scores = np.zeros(len(old_faces)); scores[::3] = .9
        before = helper.snapshot(model)
        capture = {}
        original = hb.surgery_and_stitch_params
        def stitch(*args, **kwargs):
            capture.update(keep=args[2].copy(), edges=kwargs['collapse_edges'].copy(),
                           weights=kwargs['collapse_weights'].copy())
            return original(*args, **kwargs)
        with patch.object(hb, 'surgery_and_stitch_params', side_effect=stitch):
            stats = hb.visibility_collapse_step(model, scores, .5, max_face_ratio=.1)
        self.assertGreater(stats["collapsed_edges"], 0)
        keep, edges, weights = capture['keep'], capture['edges'], capture['weights']
        remap = np.arange(len(keep)); remap[edges[:, 1]] = edges[:, 0]
        compact = np.cumsum(keep) - 1
        expected_high = compact[remap[old_faces[scores > .5]]]
        actual = set(map(tuple, model._triangle_indices.numpy()))
        self.assertTrue(all(tuple(face) in actual for face in expected_high))
        for group in model.optimizer.param_groups:
            state = model.optimizer.state[group['params'][0]]
            self.assertEqual(state['step'].item(), 7)
            for key in MOMENTS:
                old = before[group['name']][key]
                expected = old[keep].clone()
                shape = (-1,) + (1,) * (old.ndim - 1)
                expected[compact[edges[:,0]]] = (
                    old[edges[:,0]] * torch.tensor(weights[:,0], dtype=old.dtype).reshape(shape)
                    + old[edges[:,1]] * torch.tensor(weights[:,1], dtype=old.dtype).reshape(shape))
                torch.testing.assert_close(state[key], expected)
        helper.assert_can_step(model, 7)

    def test_no_op_preserves_live_parameters(self):
        helper, model = self.make()
        params = [g['params'][0] for g in model.optimizer.param_groups]
        before = helper.snapshot(model)
        for scores, ratio in ((np.ones(len(model._triangle_indices)), .1),
                              (np.zeros(len(model._triangle_indices)), .0001)):
            stats = hb.visibility_collapse_step(model, scores, .5, max_face_ratio=ratio)
            self.assertEqual(stats['collapsed_edges'], 0)
            for group, param in zip(model.optimizer.param_groups, params):
                self.assertIs(group['params'][0], param)
                for key, value in before[group['name']].items():
                    torch.testing.assert_close(model.optimizer.state[param][key], value)

    def test_no_fallback_deletion_when_topology_blocks_collapse(self):
        helper = momentum.MomentumTests(); model = helper.make_model(xyz=TETRA_X, faces=TETRA_F)
        stats = hb.visibility_collapse_step(model, np.zeros(4), .5, max_face_ratio=.9)
        self.assertEqual(stats['candidate_edges'], 6)
        self.assertEqual(stats['removed_faces'], 0)
        np.testing.assert_array_equal(model._triangle_indices.numpy(), TETRA_F)

    def test_invalid_scores_texture_and_pending_gradients_reject_before_mutation(self):
        _, model = self.make()
        faces = model._triangle_indices
        for scores in (np.zeros(1), np.full(len(faces), np.nan)):
            with self.assertRaises(ValueError): hb.visibility_collapse_step(model, scores, .5)
            self.assertIs(model._triangle_indices, faces)
        model.vertices.grad = torch.zeros_like(model.vertices)
        with self.assertRaises(RuntimeError): hb.visibility_collapse_step(model, np.zeros(len(faces)), .5)
        model.optimizer.zero_grad(set_to_none=True)
        model.face_textures = torch.zeros(len(faces), 8, 8, 3)
        with self.assertRaises(RuntimeError): hb.visibility_collapse_step(model, np.zeros(len(faces)), .5)
        self.assertIs(model._triangle_indices, faces)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA renderer required")
class CudaIntegrationTests(unittest.TestCase):
    def make(self, unused=False):
        from scene.triangle_model import TriangleModel
        helper = momentum.MomentumTests(); helper.device = 'cuda'
        xyz, faces = grid(8)
        xyz[:,:2] = (xyz[:,:2] - 3.5) * .2; xyz[:,2] = 2
        if unused: xyz = np.r_[xyz, [[9,9,9]]].astype(np.float32)
        fixture = helper.make_model(xyz=xyz, faces=faces, initialized=False, amsgrad=False)
        model = TriangleModel(3); model.__dict__.update(vars(fixture))
        model.set_sigma(.5)
        model.image_size = torch.zeros(len(faces), device='cuda')
        model.pixel_count = torch.zeros(len(faces), device='cuda', dtype=torch.int32)
        model.importance_score = torch.zeros(len(faces), device='cuda')
        return model

    def test_render_collapse_render_and_real_adam_updates(self):
        from triangle_renderer import render
        from utils.graphics_utils import getProjectionMatrix
        model = self.make()
        camera = SimpleNamespace(FoVx=np.pi/2, FoVy=np.pi/2, image_height=32, image_width=40,
            world_view_transform=torch.eye(4,device='cuda'),
            full_proj_transform=getProjectionMatrix(.01,100.,np.pi/2,np.pi/2).T.cuda(),
            camera_center=torch.zeros(3,device='cuda'))
        pipe = SimpleNamespace(debug=False, convert_SHs_python=False)
        background = torch.zeros(3,device='cuda')
        for iteration in range(3):
            out = render(camera, model, pipe, background)
            loss = (out['render'] - .3).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward(); model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
            del out, loss
            if iteration < 2:
                scores = collect_face_visibility(model, [camera], render, pipe, background)
                stats = hb.visibility_collapse_step(model, scores, .5, max_face_ratio=.1)
                self.assertGreater(stats['collapsed_edges'], 0)
                metrics = mesh_topology_metrics(model.vertices.detach().cpu().numpy(),
                                               model._triangle_indices.cpu().numpy())
                require_surface_topology(metrics)
            for group in model.optimizer.param_groups:
                p = group['params'][0]
                self.assertTrue(torch.isfinite(p).all())
                self.assertEqual(model.optimizer.state[p]['step'].item(), iteration + 1)
                self.assertEqual(model.optimizer.state[p]['exp_avg'].shape, p.shape)

    def test_actual_inline_texture_first_bake_then_upsample_without_pruning(self):
        # Execute the actual training branch on a small real model, without the
        # dataset/training loop. This catches incorrect first-phase routing.
        source = Path(__file__).resolve().parents[1] / "train.py"
        nodes = [node for node in ast.walk(ast.parse(source.read_text()))
                 if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "face_tex_phase < len(face_tex_schedule)"]
        self.assertEqual(len(nodes), 1)
        code = compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec")
        for mode in ("progressive_collapse", "none"):
            with self.subTest(mode=mode):
                model = self.make()
                before = model.vertices[model._triangle_indices.long()].detach().clone()
                scene = SimpleNamespace(getTrainCameras=Mock(side_effect=AssertionError("visibility sweep")))
                render = Mock(side_effect=AssertionError("visibility render"))
                context = dict(face_tex_phase=0, face_tex_schedule=[(28000, 4), (29000, 8)],
                    visibility_schedule=VisibilityPruningSchedule(mode=mode),
                    opt=SimpleNamespace(iterations=30000, lr_face_texture=.005, lr_face_texture_final=.0001),
                    triangles=model, scene=scene, render=render, pipe=None, background=None, torch=torch)
                with patch.object(model, 'prune_triangles', side_effect=AssertionError("face prune")):
                    for phase, (iteration, resolution) in enumerate(context['face_tex_schedule']):
                        context['iteration'] = iteration
                        exec(code, context)
                        self.assertEqual(context['face_tex_phase'], phase + 1)
                        self.assertEqual(tuple(model.face_textures.shape), (len(before), resolution, resolution, 3))
                        torch.testing.assert_close(model.vertices[model._triangle_indices.long()], before, rtol=0, atol=0)
                scene.getTrainCameras.assert_not_called()
                render.assert_not_called()

    def test_final_compaction_preserves_every_triangle_and_texture_initialization(self):
        for mode in ("progressive_collapse", "none"):
            with self.subTest(mode=mode):
                model = self.make(unused=True)
                before = model.vertices[model._triangle_indices.long()].detach().clone()
                render = Mock(side_effect=AssertionError('final visibility sweep'))
                stats = finalize_visibility_pruning(model, [], render, None, None, mode)
                self.assertEqual(stats['faces_before'], stats['faces_after'])
                torch.testing.assert_close(model.vertices[model._triangle_indices.long()], before, rtol=0, atol=0)
                self.assertEqual(len(model.vertices), 64)
                render.assert_not_called()
                model.bake_face_textures(face_tex_res=8)
                self.assertEqual(tuple(model.face_textures.shape), (len(before), 8, 8, 3))



if __name__ == '__main__': unittest.main()
