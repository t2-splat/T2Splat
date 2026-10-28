"""Runner contract tests. Maintained by zhtang2004@gmail.com."""
import contextlib
import io
import json
import os
import re
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def args_config(self, *extra):
        args = pipeline.parser().parse_args(list(extra))
        with patch.dict(os.environ, {}, clear=True):
            config = pipeline.load_config(args)
        return args, config

    def test_shell_entry_outside_project_and_spaces(self):
        with tempfile.TemporaryDirectory(prefix="t2 pipeline ") as directory:
            output = Path(directory) / "new output"
            env = os.environ.copy()
            env["PYTHON"] = sys.executable
            result = subprocess.run(["bash", str(pipeline.ROOT / "run_pipeline_mipnerf.sh"),
                                     "2", "garden", "default", "nomip", "--dry-run",
                                     "--output-root", str(output)], cwd=directory, env=env,
                                    text=True, capture_output=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["output"], str(output / "garden"))
            self.assertEqual(plan["final_iteration"], 35000)
            texture = next(s for s in plan["steps"] if s["stage"] == "texture")
            self.assertNotIn("--use_mip_texture", texture["argv"])
            self.assertFalse(output.exists())

    def test_yaml_overrides_preserve_types_and_merge_defaults(self):
        for suffix in (".yaml", ".yml"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / ("overrides" + suffix)
                path.write_text(
                    "data:\n  root: '/tmp/custom data'\n"
                    "texture:\n  face_tex_res: 16\n  use_mip_texture: false\n"
                    "  lr_face_texture: 1e-3\n"
                    "scenes:\n  custom:\n    kind: outdoor\n    images: images\n"
                    "    max_points: 123456\n")
                args, config = self.args_config("--config", str(path), "--scenes", "custom")
                self.assertEqual(config["texture"]["lr_face_texture"], .001)
                self.assertIsInstance(config["texture"]["lr_face_texture"], float)
                self.assertIs(config["texture"]["use_mip_texture"], False)
                self.assertEqual(config["training"]["iterations"], 30000)
                _, final, steps = pipeline.make_plan(args, config, "custom")
                self.assertEqual(final, 35000)
                self.assertIn("/tmp/custom data/custom", steps[0]["argv"])
                self.assertNotIn("--use_mip_texture", steps[1]["argv"])

    def test_yaml_cli_values_and_quoted_strings(self):
        _, config = self.args_config(
            "--set", "texture.lr_face_texture=1e-3",
            "--set", "texture.use_mip_texture=false",
            "--set", "training.test_iterations=[1000, 2000]",
            "--set", 'data.root="off"',
            "--set", "training.example=null")
        self.assertEqual(config["texture"]["lr_face_texture"], .001)
        self.assertIsInstance(config["texture"]["lr_face_texture"], float)
        self.assertIs(config["texture"]["use_mip_texture"], False)
        self.assertEqual(config["training"]["test_iterations"], [1000, 2000])
        self.assertEqual(config["data"]["root"], "off")
        self.assertIsNone(config["training"]["example"])

    def test_invalid_yaml_and_python_tags_are_rejected(self):
        cases = [
            ("training: [", "Invalid YAML"),
            ("[training, texture]", "must contain a mapping"),
            ("training: false", "must be a named mapping"),
            ("training: !!python/object/apply:builtins.dict []", "Invalid YAML"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            for content, message in cases:
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaisesRegex(ValueError, message):
                        self.args_config("--config", str(path))

    def test_empty_yaml_override_keeps_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.yaml"
            path.write_text("# No overrides\n")
            _, config = self.args_config("--config", str(path))
            self.assertEqual(config["texture"]["face_tex_res"], 8)
            self.assertEqual(config["training"]["iterations"], 30000)

    def test_readme_custom_yaml_runs_through_shell_entry(self):
        readme = (pipeline.ROOT / "README.md").read_text()
        section = readme.split("## Custom datasets\n", 1)[1].split("## Rendering", 1)[0]
        config = re.search(r"```yaml\n(.*?)```", section, re.S).group(1)
        command = re.search(r"```bash\n(.*?)```", section, re.S).group(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.yaml"
            for kind in ("outdoor", "indoor"):
                with self.subTest(kind=kind):
                    path.write_text(config.replace("kind: outdoor", "kind: " + kind))
                    argv = shlex.split(command.replace(chr(92) + "\n", ""))
                    argv[argv.index("--config") + 1] = str(path)
                    # Preview the documented command without starting training.
                    argv.append("--dry-run")
                    env = {"PATH": os.environ.get("PATH", ""), "PYTHON": sys.executable}
                    result = subprocess.run(argv, cwd=pipeline.ROOT, env=env,
                                            text=True, capture_output=True, check=True)
                    plan = json.loads(result.stdout)
                    self.assertEqual(plan["scene"], "my_scene")
                    self.assertEqual(plan["final_iteration"], 35000)
                    self.assertIn("/path/to/custom_dataset/my_scene", plan["steps"][0]["argv"])
                    self.assertEqual("--indoor" in plan["steps"][0]["argv"], kind == "indoor")

    def test_float_zero_is_accepted_by_legacy_integer_defaults(self):
        import argparse
        from arguments import OptimizationParams
        parser = argparse.ArgumentParser()
        OptimizationParams(parser)
        args = parser.parse_args(pipeline.cli_options({"lambda_vertex_normal": 0.0}))
        self.assertEqual(args.lambda_vertex_normal, 0)

    def test_indoor_explicit_override_wins(self):
        args, config = self.args_config("--scenes", "bonsai", "--set", "training.heartbeat_collapse_ratio=0.03")
        _, _, steps = pipeline.make_plan(args, config, "bonsai")
        argv = steps[0]["argv"]
        self.assertEqual(argv[argv.index("--heartbeat_collapse_ratio") + 1], "0.03")

    def test_intermediate_mesh_save_schedule_reaches_train_command(self):
        args, config = self.args_config(
            "--dataset", "dtu", "--set", "training.iterations=30000",
            "--set", "training.save_iterations=[20000,12000,20000]",
            "--set", "training.save_topology_snapshots=true")
        _, _, steps = pipeline.make_plan(args, config, "scan24")
        argv = steps[0]["argv"]
        start = argv.index("--save_iterations") + 1
        end = next((i for i in range(start, len(argv)) if argv[i].startswith("--")), len(argv))
        self.assertEqual(argv[start:end], ["12000", "20000", "30000"])
        self.assertIn("--save_topology_snapshots", argv)
        self.assertIn("iteration_30000", steps[0]["produces"][0])

    def test_invalid_intermediate_mesh_save_schedule_rejected(self):
        for value in ("[0]", "[30001]", "[1.5]", "[true]", "20000", "null"):
            with self.subTest(value=value):
                args, config = self.args_config("--dataset", "dtu", "--set", "training.iterations=30000",
                                               "--set", "training.save_iterations=" + value)
                with self.assertRaisesRegex(ValueError, "save_iterations"):
                    pipeline.make_plan(args, config, "scan24")

    def test_dtu_shell_all_scenes_reaches_training_parser(self):
        import argparse
        from arguments import ModelParams, OptimizationParams, PipelineParams
        parser = argparse.ArgumentParser()
        ModelParams(parser); OptimizationParams(parser); PipelineParams(parser)
        for name in ("test_iterations", "save_iterations"):
            parser.add_argument("--" + name, nargs="+", type=int)
        parser.add_argument("--save_topology_snapshots", action="store_true")
        with tempfile.TemporaryDirectory(prefix="dtu preset ") as directory:
            output = Path(directory) / "new output"
            env = {"PATH": os.environ.get("PATH", ""), "PYTHON": sys.executable}
            result = subprocess.run(["bash", str(pipeline.ROOT / "run_pipeline_dtu.sh"),
                "0", "all", "--dry-run", "--output-root", str(output)],
                cwd=directory, env=env, text=True, capture_output=True, check=True)
            plans = []; remaining = result.stdout.strip(); decoder = json.JSONDecoder()
            while remaining:
                plan, end = decoder.raw_decode(remaining)
                plans.append(plan); remaining = remaining[end:].lstrip()
            self.assertEqual({p["scene"] for p in plans}, {
                "scan24", "scan37", "scan40", "scan55", "scan63", "scan65", "scan69",
                "scan83", "scan97", "scan105", "scan106", "scan110", "scan114", "scan118", "scan122"})
            for plan in plans:
                with self.subTest(scene=plan["scene"]):
                    self.assertEqual([s["stage"] for s in plan["steps"]], ["geometry", "export", "evaluate"])
                    opts = parser.parse_args(plan["steps"][0]["argv"][3:])
                    expected = {"scan63": (1e-6, .5, .17), "scan65": (1e-5, .5, .14),
                                "scan97": (1e-5, .5, .17)}.get(plan["scene"], (1e-5, .6, .17))
                    self.assertEqual((opts.lambda_normals, opts.final_visibility_threshold,
                                      opts.prune_triangles_threshold), expected)
                    self.assertEqual((opts.iterations, opts.max_points), (20000, 650000))
                    self.assertEqual((opts.rdt_backend, opts.visibility_pruning), ("legacy", "legacy"))
                    self.assertEqual((opts.lambda_normals_super, opts.depth_lambda_init, opts.depth_lambda_final), (0, 0, 0))
                    self.assertTrue(opts.enable_heartbeat and opts.heartbeat_safety and opts.final_only_monitoring)
                    self.assertEqual(opts.test_iterations, [20000])
                    self.assertEqual(plan["output"], str(output / plan["scene"]))
            self.assertFalse(output.exists())

    def test_dtu_explicit_overrides_win_over_per_scene_settings(self):
        args, config = self.args_config("--dataset", "dtu", "--set", "training.lambda_normals=3e-5",
            "--set", "training.final_visibility_threshold=0.4", "--set", "training.prune_triangles_threshold=0.2")
        for scene in ("scan24", "scan63", "scan65", "scan97"):
            with self.subTest(scene=scene):
                _, _, steps = pipeline.make_plan(args, config, scene)
                argv = steps[0]["argv"]
                for flag, expected in [("lambda_normals", 3e-5), ("final_visibility_threshold", .4),
                                       ("prune_triangles_threshold", .2)]:
                    self.assertEqual(float(argv[argv.index("--" + flag) + 1]), expected)

    def test_invalid_final_visibility_threshold_rejected_before_training(self):
        for value in ("-0.1", "1.1", ".nan", ".inf", "true", "null", '"bad"'):
            with self.subTest(value=value):
                args, config = self.args_config("--dataset", "dtu", "--set",
                    "training.final_visibility_threshold=" + value)
                with self.assertRaisesRegex(ValueError, "final_visibility_threshold"):
                    pipeline.make_plan(args, config, "scan63")

    def test_dtu_iteration_and_texture_opt_in(self):
        args, config = self.args_config("--dataset", "dtu", "--set", "pipeline.texture=true",
                                        "--set", "pipeline.image_metrics=true")
        _, final, steps = pipeline.make_plan(args, config, "scan24")
        self.assertEqual(final, 22000)
        self.assertIn("iteration_20000", steps[1]["requires"][0])
        render = next(s for s in steps if s["stage"] == "render")
        self.assertEqual(render["argv"][render["argv"].index("--iteration") + 1], "22000")
        self.assertEqual(steps[-1]["stage"], "evaluate")

    def test_independent_texture_run_and_offload(self):
        args, config = self.args_config("--start-at", "texture", "--init-from", "output/base",
                                       "--output-root", "output/new", "--face-tex-res", "16", "--texture-offload")
        output, final, steps = pipeline.make_plan(args, config, "garden")
        self.assertEqual(output, pipeline.ROOT / "output/new/garden")
        self.assertEqual(final, 35000)
        self.assertEqual(steps[0]["stage"], "texture")
        self.assertIn(str(pipeline.ROOT / "output/base/garden"), steps[0]["argv"])
        self.assertIn("--texture_adam_offload", steps[0]["argv"])
        self.assertNotIn("geometry", [s["stage"] for s in steps])

    def test_init_from_cannot_retrain_geometry(self):
        args, config = self.args_config("--init-from", "output/base")
        with self.assertRaises(ValueError):
            pipeline.make_plan(args, config, "garden")

    def test_unknown_scene_fails_before_execution(self):
        args, config = self.args_config()
        with self.assertRaises(ValueError):
            pipeline.selected_scenes(config, "garden,typo")

    def test_tnt_and_tandt_shell_routes_are_distinct(self):
        import argparse
        from arguments import ModelParams, OptimizationParams, PipelineParams
        parser = argparse.ArgumentParser()
        ModelParams(parser); OptimizationParams(parser); PipelineParams(parser)
        for name in ("test_iterations", "save_iterations"):
            parser.add_argument("--" + name, nargs="+", type=int)
        parser.add_argument("--scene_name")
        parser.add_argument("--indoor", action="store_true")
        expected = {
            "tnt": ({"Barn", "Caterpillar", "Courthouse", "Ignatius", "Truck", "Meetingroom"},
                    ["geometry", "export", "evaluate"]),
            "tandt": ({"train", "truck"}, ["geometry", "texture", "render", "metrics", "export"]),
        }
        with tempfile.TemporaryDirectory(prefix="tnt presets ") as directory:
            for preset, (scenes, stages) in expected.items():
                output = Path(directory) / preset / "new output"
                result = subprocess.run(["bash", str(pipeline.ROOT / f"run_pipeline_{preset}.sh"),
                    "0", "all", "--dry-run", "--output-root", str(output)], cwd=directory,
                    env={"PATH": os.environ.get("PATH", ""), "PYTHON": sys.executable},
                    text=True, capture_output=True, check=True)
                plans = []; remaining = result.stdout.strip(); decoder = json.JSONDecoder()
                while remaining:
                    plan, end = decoder.raw_decode(remaining)
                    plans.append(plan); remaining = remaining[end:].lstrip()
                self.assertEqual({p["scene"] for p in plans}, scenes)
                for plan in plans:
                    with self.subTest(preset=preset, scene=plan["scene"]):
                        self.assertEqual([s["stage"] for s in plan["steps"]], stages)
                        opts = parser.parse_args(plan["steps"][0]["argv"][3:])
                        self.assertEqual(opts.iterations, 30000)
                        self.assertFalse(opts.indoor)
                        self.assertEqual((opts.lambda_normals_super, opts.depth_lambda_init, opts.depth_lambda_final), (0, 0, 0))
                        self.assertEqual((opts.rdt_backend, opts.visibility_pruning), ("surface", "progressive_collapse"))
                        self.assertTrue(opts.heartbeat_safety)
                        if preset == "tnt":
                            self.assertEqual((opts.max_points, opts.heartbeat_collapse_ratio), (5200000, .04))
                            self.assertTrue(opts.disable_priors and opts.final_only_monitoring)
                            self.assertEqual(opts.scene_name, plan["scene"])
                            self.assertEqual(opts.test_iterations, [30000])
                            self.assertEqual(plan["final_iteration"], 30000)
                            self.assertEqual(opts.source_path, str(pipeline.ROOT.parent / "datasets/TNT" / plan["scene"] / plan["scene"]))
                            self.assertNotIn("--clean", plan["steps"][1]["argv"])
                            self.assertEqual(len(plan["steps"][2]["produces"]), 4)
                        else:
                            self.assertEqual((opts.max_points, opts.heartbeat_collapse_ratio), (4000000, .08))
                            self.assertFalse(opts.disable_priors)
                            self.assertEqual(plan["final_iteration"], 35000)
                            tex = plan["steps"][1]["argv"]
                            self.assertEqual(tex[tex.index("--face_tex_res") + 1], "8")
                            self.assertIn("--use_mip_texture", tex)
                self.assertFalse(output.exists())

    def test_tnt_prior_defaults_and_explicit_overrides(self):
        args, config = self.args_config("--dataset", "tnt", "--reg-mode", "strong")
        _, _, steps = pipeline.make_plan(args, config, "Meetingroom")
        argv = steps[0]["argv"]
        self.assertIn("--disable_priors", argv)
        for flag in ["--lambda_normals_super", "--depth_lambda_init", "--depth_lambda_final"]:
            self.assertEqual(float(argv[argv.index(flag)+1]), 0)
        args, config = self.args_config("--dataset", "tnt", "--set", "training.disable_priors=false",
            "--set", "training.lambda_normals_super=0.01", "--set", "training.heartbeat_collapse_ratio=0.06")
        _, _, steps = pipeline.make_plan(args, config, "Meetingroom")
        argv = steps[0]["argv"]
        self.assertNotIn("--disable_priors", argv)
        self.assertEqual(argv[argv.index("--lambda_normals_super")+1], "0.01")
        self.assertEqual(argv[argv.index("--heartbeat_collapse_ratio")+1], "0.06")

    def test_tnt_geometry_metrics_reject_stale_protocol_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); mesh = root / "mesh.ply"; ev = root / "evaluation"; ev.mkdir()
            mesh.write_text("ply\nformat ascii 1.0\nelement vertex 3\nelement face 1\nend_header\n")
            (ev / "Barn.prf_tau_plotstr.txt").write_text("0.4 0.6 0.48 0.01 5\n")
            metadata = dict(status="complete", protocol="virtual_scan_scale_corrected_v1", scene="Barn",
                            crop_prediction_and_ground_truth=True, evaluation_resolution=2,
                            mesh_path=str(mesh), source_path=str(root / "input"), mesh_vertices=3, mesh_faces=1,
                            precision=.4, recall=.6, fscore=.48, tau_gt_units=.01)
            path = ev / "evaluation_metadata.json"; path.write_text(json.dumps(metadata))
            step = dict(argv=[sys.executable, "-u", "virtual_scan_sampling_eval.py", "--out-dir", str(ev),
                              "--dataset-dir", str(root / "gt/Barn"), "--ply-path", str(mesh), "-s", str(root / "input"), "-r", "2"])
            result = pipeline.tnt_geometry_metrics(step)
            self.assertEqual((result["fscore"], result["mesh_faces"]), (.48, 1))
            for field, bad in [("protocol", "virtual_scan"), ("mesh_faces", 2),
                               ("crop_prediction_and_ground_truth", False), ("fscore", .49),
                               ("source_path", str(root / "wrong"))]:
                with self.subTest(field=field):
                    path.write_text(json.dumps({**metadata, field: bad}))
                    with self.assertRaises(ValueError): pipeline.tnt_geometry_metrics(step)
            path.write_text(json.dumps(metadata))
            (ev / "Barn.prf_tau_plotstr.txt").write_text("0.4 0.6 nan 0.01 5\n")
            with self.assertRaises(ValueError): pipeline.tnt_geometry_metrics(step)

    def run_fixture(self, directory, code, produces=(), extra_steps=()):
        args, config = self.args_config()
        output = Path(directory) / "run"
        plan = [{"stage": "geometry", "argv": [sys.executable, "-c", code], "cwd": directory,
                 "requires": [], "produces": list(map(str, produces))}, *extra_steps]
        with patch.object(pipeline, "make_plan", return_value=(output, 30000, plan)), contextlib.redirect_stdout(io.StringIO()):
            pipeline.execute_scene(args, config, "garden")
        return output

    def test_subprocess_failure_stops_later_stages_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must_not_exist"
            later = {"stage": "texture", "argv": [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('bad')"],
                     "cwd": directory, "requires": [], "produces": []}
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_fixture(directory, "raise SystemExit(7)", extra_steps=[later])
            status = json.loads((Path(directory) / "run/pipeline_status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["completed_stages"], [])
            self.assertFalse(marker.exists())

    def test_successful_exit_without_artifact_is_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "missing or empty output"):
                self.run_fixture(directory, "print('finished')", [Path(directory) / "missing.pt"])
            status = json.loads((Path(directory) / "run/pipeline_status.json").read_text())
            self.assertEqual(status["status"], "failed")

    def test_success_records_completion_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.pt"
            output = self.run_fixture(directory, f"open({str(artifact)!r}, 'w').write('fixture')", [artifact])
            status = json.loads((output / "pipeline_status.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["completed_stages"], ["geometry"])

    def test_existing_geometry_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "run/point_cloud").mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                self.run_fixture(directory, "raise SystemExit('must not execute')")


if __name__ == "__main__":
    unittest.main()
