"""COLMAP no-prior mode must keep observations and splits while ignoring priors."""
import contextlib
import io
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from arguments import ModelParams
import scene
from scene import dataset_readers as readers


class NoPriorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        sparse = self.root / 'sparse/0'
        sparse.mkdir(parents=True)
        (self.root / 'images').mkdir()
        (self.root / 'mask').mkdir()
        (sparse / 'cameras.txt').write_text('1 PINHOLE 8 6 10 10 4 3\n')
        records = []
        for index in reversed(range(4)):
            name = f'{index:04d}'
            Image.new('RGB', (8, 6), (index * 30, 70, 100)).save(
                self.root / 'images' / f'{name}.png')
            Image.new('L', (8, 6), 255).save(self.root / 'mask' / f'{name}.png')
            records.append(f'{index + 1} 1 0 0 0 {index} 0 0 1 {name}.png\n0 0 -1\n')
        (sparse / 'images.txt').write_text(''.join(records))
        readers.storePly(str(sparse / 'points3D.ply'),
                         np.array([[0., 1., 2.], [1., 2., 3.]]),
                         np.array([[20, 30, 40], [50, 60, 70]]))

    def add_priors(self):
        (self.root / 'normals').mkdir()
        (self.root / 'depth').mkdir()
        params = {}
        for index in range(4):
            name = f'{index:04d}'
            Image.new('RGB', (8, 6), (128, 64, 255)).save(
                self.root / 'normals' / f'{name}.png')
            Image.new('I;16', (8, 6), 100).save(self.root / 'depth' / f'{name}.png')
            params[name] = {'scale': 2., 'offset': .25}
        (self.root / 'sparse/0/depth_params.json').write_text(json.dumps(params))

    def read(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            info = readers.readColmapSceneInfo(str(self.root), 'images', True,
                                               llffhold=2, **kwargs)
        for camera in info.train_cameras + info.test_cameras:
            self.addCleanup(camera.image.close)
        return info

    def test_default_loads_priors_and_disable_keeps_images_masks_cameras_and_point_cloud(self):
        self.add_priors()
        enabled = self.read()
        disabled = self.read(disable_priors=True)
        self.assertEqual([c.image_name for c in disabled.train_cameras], ['0001', '0003'])
        self.assertEqual([c.image_name for c in disabled.test_cameras], ['0000', '0002'])
        for old, new in zip(enabled.train_cameras + enabled.test_cameras,
                            disabled.train_cameras + disabled.test_cameras):
            self.assertIsNotNone(old.normal_map)
            self.assertEqual(old.depth_params, {'scale': 2., 'offset': .25, 'med_scale': 2.})
            self.assertTrue(Path(old.depth_path).is_file())
            self.assertIsNone(new.normal_map)
            self.assertIsNone(new.depth_params)
            self.assertEqual(new.depth_path, '')
            self.assertEqual(old.image_path, new.image_path)
            self.assertEqual(old.mask_path, new.mask_path)
            self.assertTrue(Path(new.mask_path).is_file())
            self.assertEqual((old.uid, old.FovX, old.FovY, old.width, old.height),
                             (new.uid, new.FovX, new.FovY, new.width, new.height))
            for first, second in ((old.R, new.R), (old.T, new.T),
                                  (np.array(old.image), np.array(new.image))):
                np.testing.assert_array_equal(first, second)
        self.assertEqual(enabled.ply_path, disabled.ply_path)
        np.testing.assert_array_equal(enabled.point_cloud.points, disabled.point_cloud.points)
        np.testing.assert_array_equal(enabled.point_cloud.colors, disabled.point_cloud.colors)
        np.testing.assert_array_equal(enabled.nerf_normalization['translate'],
                                      disabled.nerf_normalization['translate'])
        self.assertEqual(enabled.nerf_normalization['radius'], disabled.nerf_normalization['radius'])

    def test_disabled_ignores_corrupt_depth_metadata_and_prior_images(self):
        self.add_priors()
        (self.root / 'sparse/0/depth_params.json').write_text('invalid json')
        for folder in ('normals', 'depth'):
            for path in (self.root / folder).glob('*.png'):
                path.write_bytes(b'invalid image')
        info = self.read(disable_priors=True)
        self.assertEqual(len(info.train_cameras) + len(info.test_cameras), 4)
        for camera in info.train_cameras + info.test_cameras:
            self.assertEqual(camera.image.size, (8, 6))
            self.assertIsNone(camera.normal_map)
            self.assertIsNone(camera.depth_params)
            self.assertEqual(camera.depth_path, '')
        with self.assertRaises(SystemExit):
            self.read()

    def test_disabled_does_not_create_missing_prior_directories(self):
        info = self.read(disable_priors=True)
        self.assertEqual(len(info.train_cameras), 2)
        self.assertFalse((self.root / 'normals').exists())
        self.assertFalse((self.root / 'depth').exists())
        default = self.read()
        self.assertTrue((self.root / 'normals').is_dir())
        self.assertTrue(all(c.normal_map is None for c in default.train_cameras))

    def test_argument_defaults_to_enabled_prior_loading_and_opt_in_is_extractable(self):
        parser = ArgumentParser()
        model = ModelParams(parser)
        self.assertFalse(model.extract(parser.parse_args([])).disable_priors)
        self.assertTrue(model.extract(parser.parse_args(['--disable_priors'])).disable_priors)

    def test_scene_passes_flag_and_accepts_older_args_without_it(self):
        info = self.read(disable_priors=True)
        for flag in (True, None):
            args = Namespace(source_path=str(self.root), images='images', eval=True,
                             model_path=str(self.root / 'model'))
            if flag is not None:
                args.disable_priors = flag
            loader = Mock(return_value=info)
            with self.subTest(flag=flag), \
                    patch.dict(scene.sceneLoadTypeCallbacks, {'Colmap': loader}), \
                    patch.object(scene, 'cameraList_from_camInfos', side_effect=lambda cameras, *a: cameras), \
                    contextlib.redirect_stdout(io.StringIO()):
                scene.Scene(args, Mock(), .1, .5, load_iteration=1, shuffle=False)
            loader.assert_called_once_with(str(self.root), 'images', True,
                                           disable_priors=bool(flag))


if __name__ == '__main__':
    unittest.main()
