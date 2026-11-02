"""CPU regressions for DTU G selection, candidate routing and training RNG."""
from argparse import ArgumentParser
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from arguments import OptimizationParams
from utils.rdt_g import select_faces
from utils.rdt_surface import run_g_surface_rdt, select_manifold_faces


class GSelectorTests(unittest.TestCase):
    def select(self, faces, scores=None):
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        scores = np.ones(len(faces)) if scores is None else np.asarray(scores)
        originals = faces.copy(), scores.copy()
        out, ids, stats, flips = select_faces(
            faces, int(faces.max()) + 1 if faces.size else 0, scores,
            max_edge_faces=4, enforce_vertex_fan=False, enforce_orientation=False)
        np.testing.assert_array_equal(out, faces[ids])
        np.testing.assert_array_equal(faces, originals[0])
        np.testing.assert_array_equal(scores, originals[1])
        self.assertEqual(out.dtype, np.dtype('int64'))
        self.assertEqual(ids.dtype, np.dtype('int64'))
        self.assertTrue(np.all(ids[1:] > ids[:-1]))
        self.assertFalse(flips.any())
        for key in ('flipped_faces', 'computed_flipped_faces', 'orientation_rejected', 'fan_rejected'):
            self.assertEqual(stats[key], 0)
        return out, ids, stats

    def test_edge_cap_four_keeps_candidate_order_and_winding(self):
        faces = [[0, 1, i] for i in range(2, 7)]
        out, ids, stats = self.select(faces, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(ids, [1, 2, 3, 4])
        self.assertEqual(stats['edge_rejected'], 1)
        self.assertEqual(len(out), 4)
        # The pre-existing strict selector still admits at most two edge faces.
        self.assertEqual(len(select_manifold_faces(np.array(faces), 7)[0]), 2)

    def test_reverse_and_cyclic_duplicates_rejected(self):
        out, ids, stats = self.select([[0, 1, 2], [2, 1, 0], [1, 2, 0]], [1, 3, 2])
        np.testing.assert_array_equal(ids, [1])
        np.testing.assert_array_equal(out, [[2, 1, 0]])
        self.assertEqual(stats['duplicate_rejected'], 2)

    def test_disconnected_vertex_fans_retained(self):
        faces = np.array([[0, 1, 2], [0, 3, 4]])
        self.assertEqual(len(self.select(faces)[0]), 2)
        self.assertEqual(len(select_manifold_faces(faces, 5)[0]), 1)

    def test_incoherent_tetrahedron_winding_unchanged(self):
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        np.testing.assert_array_equal(self.select(faces)[0], faces)
        strict, _, stats = select_manifold_faces(faces, 4)
        self.assertEqual(len(strict), 4)
        self.assertGreater(stats['flipped_faces'], 0)

    def test_ineligible_faces_and_empty_input(self):
        faces = [[0, 1, 2], [0, 3, 4], [1, 1, 2], [5, 6, 7]]
        _, ids, stats = self.select(faces, [1, np.nan, 2, -np.inf])
        np.testing.assert_array_equal(ids, [0])
        self.assertEqual(stats['ineligible_faces'], 3)
        self.assertEqual(self.select([])[0].shape, (0, 3))

    def test_frontier_growth_precedes_disconnected_higher_score_seed(self):
        # Candidate 1 reaches edge (2,3) through candidate 0. It must be tried
        # before candidates 2..5, even though those have higher global scores.
        faces = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [2, 3, 5], [2, 3, 6], [2, 3, 7]]
        _, ids, _ = self.select(faces, [10, 1, 9, 8, 7, 6])
        np.testing.assert_array_equal(ids, [0, 1, 2, 3, 4])


class GCandidatePathTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.zeros((7, 3), dtype=np.float32)
        self.source = np.array([[0, 1, 2]], dtype=np.int32)
        self.candidates = np.array([[0, 1, i] for i in range(2, 7)], dtype=np.int64)

    def run_g(self, generator, **kwargs):
        with redirect_stdout(io.StringIO()):
            return run_g_surface_rdt(self.vertices, self.source, rdt_run=generator, **kwargs)

    def test_external_selection_uses_unfiltered_legacy_candidates(self):
        for debug_enabled in (False, True):
            with self.subTest(return_debug=debug_enabled):
                debug = dict(input_faces=self.source, candidate_faces=np.array([[2, 3, 4]]),
                             selected_faces=self.candidates, rejected_faces=np.array([[2, 3, 4]]),
                             selector_stats=None, output_faces=self.candidates)
                generator = Mock(return_value=(self.candidates, debug) if debug_enabled else self.candidates)
                scorer = Mock(return_value=(np.array([1., 2., 3., 4., 5.]), {'test': True}))
                with patch('utils.rdt_surface.score_surface_candidates', scorer), \
                     patch('utils.rdt_surface.select_manifold_faces', side_effect=AssertionError('strict selector used')):
                    result = self.run_g(generator, return_debug=debug_enabled)
                self.assertEqual(generator.call_count, 1)
                self.assertEqual(generator.call_args.kwargs, dict(verbose=False, orient=False,
                                 return_debug=debug_enabled, face_selector=None))
                self.assertIs(scorer.call_args.args[2], self.candidates)
                self.assertEqual(scorer.call_args.kwargs, dict(max_distance_ratio=.5,
                                 min_normal_cos=.15, min_quality=.02))
                out = result[0] if debug_enabled else result
                np.testing.assert_array_equal(out, self.candidates[1:])
                if debug_enabled:
                    info = result[1]
                    self.assertIs(info['selected_faces'], self.candidates)
                    self.assertIs(info['candidate_faces'], debug['candidate_faces'])
                    np.testing.assert_array_equal(info['output_faces'], out)
                    self.assertEqual(info['selector_stats']['max_edge_faces'], 4)
                    self.assertIsNone(debug['selector_stats'])

    def test_rng_restored_after_generation_scoring_and_failure(self):
        def consume():
            random.random(); np.random.random(5); torch.rand(5)

        def generate(*args, **kwargs):
            consume()
            return self.candidates

        python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        for fail_at in (None, 'generation', 'scoring', 'empty'):
            with self.subTest(fail_at=fail_at):
                def generator(*args, **kwargs):
                    result = generate()
                    if fail_at == 'generation':
                        raise RuntimeError('generation failed')
                    return result

                def score(*args, **kwargs):
                    consume()
                    if fail_at == 'scoring':
                        raise RuntimeError('scoring failed')
                    return np.full(5, -np.inf if fail_at == 'empty' else 1.), {}

                with patch('utils.rdt_surface.score_surface_candidates', side_effect=score):
                    if fail_at:
                        with self.assertRaises(RuntimeError):
                            self.run_g(generator)
                    else:
                        self.run_g(generator)
                self.assertEqual(random.getstate(), python_before)
                after = np.random.get_state()
                self.assertEqual(after[0], numpy_before[0])
                np.testing.assert_array_equal(after[1], numpy_before[1])
                self.assertEqual(after[2:], numpy_before[2:])
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))

    def test_initialized_cuda_rng_streams_are_restored(self):
        states = [torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([3, 4], dtype=torch.uint8)]
        with patch('torch.cuda.is_initialized', return_value=True), \
             patch('torch.cuda.get_rng_state_all', return_value=states), \
             patch('torch.cuda.set_rng_state_all') as restore, \
             patch('utils.rdt_surface.score_surface_candidates', return_value=(np.ones(5), {})):
            self.run_g(Mock(return_value=self.candidates))
        restore.assert_called_once()
        for actual, expected in zip(restore.call_args.args[0], states):
            self.assertIsNot(actual, expected)
            self.assertTrue(torch.equal(actual, expected))


class GTrainingRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the real method while avoiding scene imports and CUDA kernels.
        path = Path(__file__).resolve().parents[1] / 'scene/triangle_model.py'
        tree = ast.parse(path.read_text())
        cls.method = next(node for node in ast.walk(tree)
                          if isinstance(node, ast.FunctionDef) and node.name == 'run_restricted_delaunay')
        cls.code = compile(ast.Module(body=[cls.method], type_ignores=[]), str(path), 'exec')

    def invoke(self, backend='surface', **kwargs):
        candidates = np.array([[0, 1, 2]], dtype=np.int64)
        model = SimpleNamespace(vertices=torch.zeros(3, 3), _triangle_indices=torch.tensor(candidates))
        generator = Mock(return_value=candidates)
        torch_cpu = SimpleNamespace(
            as_tensor=lambda value, **kw: torch.as_tensor(value, device='cpu'),
            zeros=lambda shape, **kw: torch.zeros(shape, dtype=kw['dtype']),
            float=torch.float, int=torch.int, int32=torch.int32)
        context = dict(np=np, torch=torch_cpu, rdel=SimpleNamespace(run=generator))
        exec(self.code, context)
        with redirect_stdout(io.StringIO()), \
             patch('utils.rdt_surface.run_g_surface_rdt', return_value=candidates) as g_path:
            context['run_restricted_delaunay'](model, backend=backend, **kwargs)
        return model, generator, g_path

    def test_default_manifold_and_legacy_routes_unchanged(self):
        _, generator, g_path = self.invoke()
        g_path.assert_not_called()
        callback = generator.call_args.kwargs['face_selector']
        self.assertEqual(callback.func.__name__, 'select_rdt_surface')
        self.assertEqual(callback.keywords, dict(max_distance_ratio=.5, min_normal_cos=.15, min_quality=.02))
        _, generator, g_path = self.invoke(backend='legacy')
        g_path.assert_not_called()
        self.assertIsNone(generator.call_args.kwargs['face_selector'])

    def test_g_routes_to_external_selector(self):
        model, generator, g_path = self.invoke(surface_mode='g')
        generator.assert_not_called()
        g_path.assert_called_once()
        self.assertIs(g_path.call_args.kwargs['rdt_run'], generator)
        np.testing.assert_array_equal(model._triangle_indices.numpy(), [[0, 1, 2]])

    def test_invalid_surface_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown RDT surface mode'):
            self.invoke(surface_mode='unknown')

    def test_cli_default_and_g_option(self):
        parser = ArgumentParser()
        OptimizationParams(parser)
        self.assertEqual(parser.parse_args([]).rdt_surface_mode, 'manifold')
        self.assertEqual(parser.parse_args(['--rdt_surface_mode', 'g']).rdt_surface_mode, 'g')


if __name__ == '__main__':
    unittest.main()
