#
# The original code is under the following copyright:
# Copyright (C) 2022, Google LLC
# Licensed under the Apache License, Version 2.0
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import numpy as np
import os
from typing import List, Mapping, Optional, Text, Tuple, Union
import copy
from PIL import Image
import mediapy as media
from matplotlib import cm
from tqdm import tqdm
from utils.graphics_utils import getProjectionMatrix

import torch

def normalize(x: np.ndarray) -> np.ndarray:
	"""Normalization helper function."""
	return x / np.linalg.norm(x)

def pad_poses(p: np.ndarray) -> np.ndarray:
	"""Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
	bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
	return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
	"""Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
	return p[..., :3, :4]


def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""Recenter poses around the origin."""
	cam2world = average_pose(poses)
	transform = np.linalg.inv(pad_poses(cam2world))
	poses = transform @ pad_poses(poses)
	return unpad_poses(poses), transform


def average_pose(poses: np.ndarray) -> np.ndarray:
	"""New pose using average position, z-axis, and up vector of input poses."""
	position = poses[:, :3, 3].mean(0)
	z_axis = poses[:, :3, 2].mean(0)
	up = poses[:, :3, 1].mean(0)
	cam2world = viewmatrix(z_axis, up, position)
	return cam2world

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
              position: np.ndarray) -> np.ndarray:
	"""Construct lookat view matrix."""
	vec2 = normalize(lookdir)
	vec0 = normalize(np.cross(up, vec2))
	vec1 = normalize(np.cross(vec2, vec0))
	m = np.stack([vec0, vec1, vec2, position], axis=1)
	return m

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
	"""Calculate nearest point to all focal axes in poses."""
	directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
	m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
	mt_m = np.transpose(m, [0, 2, 1]) @ m
	focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
	return focus_pt


def orbit_lookat_point(poses_34: np.ndarray, mode: str = "cameras_mean") -> np.ndarray:
	"""Pick the world-space point that orbit cameras look at.

	``focus_point_fn`` (rays) follows mip-NeRF / NeRF-style least-squares ray intersection.
	For inward-looking 360° captures it is often fine, but can land oddly off-center or
	be numerically brittle when rays are near-parallel or asymmetric. ``cameras_mean``
	uses the centroid of camera centers — usually a better \"scene center\" for MipNeRF-360.
	"""
	centers = poses_34[:, :3, 3]
	mean_c = np.mean(centers, axis=0)
	m = (mode or "cameras_mean").lower().replace("-", "_")
	if m in ("rays", "intersect", "focus_point"):
		try:
			return focus_point_fn(poses_34)
		except np.linalg.LinAlgError:
			return mean_c
	if m in ("hybrid", "blend"):
		try:
			fp = focus_point_fn(poses_34)
			return 0.5 * (fp + mean_c)
		except np.linalg.LinAlgError:
			return mean_c
	# cameras_mean, mean, default
	return mean_c

def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""Transforms poses so principal components lie on XYZ axes.

	Args:
	poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

	Returns:
	A tuple (poses, transform), with the transformed poses and the applied
	camera_to_world transforms.
	"""
	t = poses[:, :3, 3]
	t_mean = t.mean(axis=0)
	t = t - t_mean

	eigval, eigvec = np.linalg.eig(t.T @ t)
	# Sort eigenvectors in order of largest to smallest eigenvalue.
	inds = np.argsort(eigval)[::-1]
	eigvec = eigvec[:, inds]
	rot = eigvec.T
	if np.linalg.det(rot) < 0:
		rot = np.diag(np.array([1, 1, -1])) @ rot

	transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
	poses_recentered = unpad_poses(transform @ pad_poses(poses))
	transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

	# Flip coordinate system if z component of y-axis is negative
	if poses_recentered.mean(axis=0)[2, 1] < 0:
		poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
		transform = np.diag(np.array([1, -1, -1, 1])) @ transform

	return poses_recentered, transform
	# points = np.random.rand(3,100)
	# points_h = np.concatenate((points,np.ones_like(points[:1])), axis=0)
	# (poses_recentered @ points_h)[0]
	# (transform @ pad_poses(poses) @ points_h)[0,:3]
	# import pdb; pdb.set_trace()

	# # Just make sure it's it in the [-1, 1]^3 cube
	# scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
	# poses_recentered[:, :3, 3] *= scale_factor
	# transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

	# return poses_recentered, transform

def generate_ellipse_path(poses: np.ndarray,
                        n_frames: int = 120,
                        const_speed: bool = True,
                        z_variation: float = 0.,
                        z_phase: float = 0.) -> np.ndarray:
	"""Generate an elliptical render path based on the given poses."""
	# Calculate the focal point for the path (cameras point toward this).
	center = focus_point_fn(poses)
	# Path height sits at z=0 (in middle of zero-mean capture pattern).
	offset = np.array([center[0], center[1], 0])

	# Calculate scaling for ellipse axes based on input camera positions.
	sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
	# Use ellipse that is symmetric about the focal point in xy.
	low = -sc + offset
	high = sc + offset
	# Optional height variation need not be symmetric
	z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
	z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

	def get_positions(theta):
		# Interpolate between bounds with trig functions to get ellipse in x-y.
		# Optionally also interpolate in z to change camera height along path.
		return np.stack([
			low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
			low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
			z_variation * (z_low[2] + (z_high - z_low)[2] *
							(np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
		], -1)

	theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
	positions = get_positions(theta)

	#if const_speed:

	# # Resample theta angles so that the velocity is closer to constant.
	# lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
	# theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
	# positions = get_positions(theta)

	# Throw away duplicated last position.
	positions = positions[:-1]

	# Set path's up vector to axis closest to average of input pose up vectors.
	avg_up = poses[:, :3, 1].mean(0)
	avg_up = avg_up / np.linalg.norm(avg_up)
	ind_up = np.argmax(np.abs(avg_up))
	up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

	return np.stack([viewmatrix(p - center, up, p) for p in positions])


def generate_path(viewpoint_cameras, n_frames=480):
	c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_cameras])
	pose = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
	pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

	# generate new poses
	new_poses = generate_ellipse_path(poses=pose_recenter, n_frames=n_frames)
	# warp back to orignal scale
	new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

	traj = []
	for c2w in new_poses:
		c2w = c2w @ np.diag([1, -1, -1, 1])
		cam = copy.deepcopy(viewpoint_cameras[0])
		cam.image_height = int(cam.image_height / 2) * 2
		cam.image_width = int(cam.image_width / 2) * 2
		cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
		cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
		cam.camera_center = cam.world_view_transform.inverse()[3, :3]
		traj.append(cam)

	return traj


def generate_smooth_orbit_path(
	viewpoint_cameras,
	n_frames=480,
	radius_scale=1.0,
	height_frac=0.08,
	ease="smoothstep",
	loops=1.0,
	focus_mode="cameras_mean",
):
	"""Smooth horizontal orbit around a scene-centered look-at point.

	Unlike :func:`generate_path`, this avoids PCA + ellipse warping (often uneven speed
	and odd tilting). Camera motion uses eased angular sampling; ``focus_mode`` sets
	the look-at target — see :func:`orbit_lookat_point`.
	"""
	if not viewpoint_cameras:
		return []

	def _c2w_from_cam(cam):
		wvt = cam.world_view_transform.cpu().numpy()
		return np.linalg.inv(wvt.T)

	c2ws = np.stack([_c2w_from_cam(c) for c in viewpoint_cameras], axis=0)
	poses_34 = c2ws[:, :3, :].copy()
	poses_34 = poses_34 @ np.diag([1, -1, -1, 1])

	centers = poses_34[:, :3, 3]
	focal = orbit_lookat_point(poses_34, mode=focus_mode)

	dists = np.linalg.norm(centers - focal, axis=1)
	radius = float(np.percentile(dists, 60)) * float(radius_scale)
	radius = max(radius, 1e-3)

	mean_up = poses_34[:, :3, 1].mean(axis=0)
	nu = np.linalg.norm(mean_up)
	if nu < 1e-6:
		mean_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
	else:
		mean_up = mean_up / nu

	ref = centers[0] - focal
	ref = ref - mean_up * np.dot(ref, mean_up)
	nr = np.linalg.norm(ref)
	if nr < 1e-6:
		ref = np.cross(mean_up, np.array([1.0, 0.0, 0.0]))
		nr = np.linalg.norm(ref)
		if nr < 1e-6:
			ref = np.cross(mean_up, np.array([0.0, 0.0, 1.0]))
		ref = ref / max(np.linalg.norm(ref), 1e-9)
	else:
		ref = ref / nr

	u = np.cross(mean_up, ref)
	u = u / max(np.linalg.norm(u), 1e-9)
	v = np.cross(u, mean_up)
	v = v / max(np.linalg.norm(v), 1e-9)

	traj = []
	n = int(max(1, n_frames))
	ease = (ease or "smoothstep").lower()
	loops = float(loops if loops is not None else 1.0)

	for i in range(n):
		t_lin = i / max(n - 1, 1)
		if ease == "linear":
			te = t_lin
		elif ease == "smootherstep":
			te = t_lin * t_lin * t_lin * (t_lin * (t_lin * 6.0 - 15.0) + 10.0)
		else:
			te = t_lin * t_lin * (3.0 - 2.0 * t_lin)

		phi = 2.0 * np.pi * loops * te
		pos = focal + radius * (np.cos(phi) * u + np.sin(phi) * v)
		if height_frac:
			pos = pos + mean_up * (height_frac * radius * np.sin(phi * 2.0))

		lookdir = focal - pos
		ld = np.linalg.norm(lookdir)
		if ld < 1e-9:
			lookdir = mean_up
		else:
			lookdir = lookdir / ld

		pose_34 = viewmatrix(lookdir, mean_up, pos)
		c2w = pad_poses(pose_34) @ np.diag([1, -1, -1, 1])

		cam = copy.deepcopy(viewpoint_cameras[0])
		cam.image_height = int(cam.image_height / 2) * 2
		cam.image_width = int(cam.image_width / 2) * 2
		cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
		cam.full_proj_transform = (
			cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))
		).squeeze(0)
		cam.camera_center = cam.world_view_transform.inverse()[3, :3]
		traj.append(cam)

	return traj


def generate_zoom_trajectory(viewpoint_cameras, n_frames=480, zoom_start=0, zoom_duration=120, zoom_intensity=2.0):
    traj = generate_path(viewpoint_cameras, n_frames=n_frames)

    cam0 = viewpoint_cameras[0]
    orig_fovx = cam0.FoVx
    orig_fovy = cam0.FoVy
    orig_focalx = cam0.image_width / (2 * np.tan(orig_fovx / 2))
    orig_focaly = cam0.image_height / (2 * np.tan(orig_fovy / 2))

    for i, cam in enumerate(traj):
        cam = copy.deepcopy(cam)

        if zoom_start <= i < zoom_start + zoom_duration:
            t = (i - zoom_start) / max(zoom_duration - 1, 1)
            zoom_factor = 1 + t * (zoom_intensity - 1)

        elif zoom_start + zoom_duration <= i < zoom_start + 2 * zoom_duration:
            t = (i - (zoom_start + zoom_duration)) / max(zoom_duration - 1, 1)
            zoom_factor = zoom_intensity - t * (zoom_intensity - 1)
        else:
            zoom_factor = 1.0

        new_focalx = orig_focalx * zoom_factor
        new_focaly = orig_focaly * zoom_factor
        new_fovx = 2 * np.arctan(cam.image_width / (2 * new_focalx))
        new_fovy = 2 * np.arctan(cam.image_height / (2 * new_focaly))
        cam.FoVx = new_fovx
        cam.FoVy = new_fovy

        cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=new_fovx, fovY=new_fovy).transpose(0,1).cuda()
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        traj[i] = cam
    return traj

def load_img(pth: str) -> np.ndarray:
	"""Load an image and cast to float32."""
	with open(pth, 'rb') as f:
		image = np.array(Image.open(f), dtype=np.float32)
	return image


def create_videos(base_dir, input_dir, out_name, num_frames=480):
	"""Creates videos out of the images saved to disk."""
	# Last two parts of checkpoint path are experiment name and scene name.
	video_prefix = f'{out_name}'

	zpad = max(5, len(str(num_frames - 1)))
	idx_to_str = lambda idx: str(idx).zfill(zpad)

	os.makedirs(base_dir, exist_ok=True)

	img_file = os.path.join(input_dir, 'renders', f'{idx_to_str(0)}.png')
	img = load_img(img_file)
	shape = img.shape

	print(f'Video shape is {shape[:2]}')

	video_kwargs = {
		'shape': shape[:2],
		'codec': 'h264',
		'fps': 120,
		'crf': 18,
	}

	video_file = os.path.join(base_dir, f'{video_prefix}_color.mp4')
	input_format = 'rgb'

	file_ext = 'png'
	idx = 0

	file0 = os.path.join(input_dir, 'renders', f'{idx_to_str(0)}.{file_ext}')

	if not os.path.exists(file0):
		return
	print(f'Making video {video_file}...')
	with media.VideoWriter(video_file, **video_kwargs, input_format=input_format) as writer:
		for idx in tqdm(range(num_frames)):

			img_file = os.path.join(input_dir, 'renders', f'{idx_to_str(idx)}.{file_ext}')

			if not os.path.exists(img_file):
				ValueError(f'Image file {img_file} does not exist.')
			img = load_img(img_file)
			img = img / 255.

			frame = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
			writer.add_image(frame)
			idx += 1

def save_img_u8(img, pth):
	"""Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
	with open(pth, 'wb') as f:
		Image.fromarray(
			(np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
				f, 'PNG')
		
def save_img_f32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open(pth, 'wb') as f:
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')


def write_mesh_evolution_video(frames_dir: str, output_mp4: str, fps: float = 24.0) -> bool:
	"""Encode mesh evolution PNGs (including optional RDT stage frames) into one MP4."""
	import glob
	import re

	if not os.path.isdir(frames_dir):
		print(f"[mesh_viz] frames dir missing: {frames_dir}")
		return False

	numbered = []
	for p in glob.glob(os.path.join(frames_dir, "*.png")):
		base = os.path.basename(p)
		entry = None
		m = re.match(r"^frame_(\d+)\.png$", base)
		if m:
			entry = (int(m.group(1)), 0, 0, p)
		m = re.match(r"^rdt_before_(\d+)\.png$", base)
		if m:
			entry = (int(m.group(1)), 10, 0, p)
		m = re.match(r"^rdt_candprog_(\d+)_(\d+)\.png$", base)
		if m:
			entry = (int(m.group(1)), 11, int(m.group(2)), p)
		m = re.match(r"^rdt_candidate_(\d+)\.png$", base)
		if m:
			entry = (int(m.group(1)), 30, 0, p)
		m = re.match(r"^rdt_rejected_(\d+)\.png$", base)
		if m:
			entry = (int(m.group(1)), 40, 0, p)
		m = re.match(r"^rdt_selected_(\d+)\.png$", base)
		if m:
			entry = (int(m.group(1)), 50, 0, p)
		m = re.match(r"^rdt_after_(\d+)\.png$", base)
		if m:
			entry = (int(m.group(1)), 60, 0, p)
		if entry is not None:
			numbered.append(entry)
	numbered.sort(key=lambda t: (t[0], t[1], t[2]))
	paths = [t[3] for t in numbered]
	final_p = os.path.join(frames_dir, "frame_final.png")
	if os.path.isfile(final_p):
		paths.append(final_p)

	if not paths:
		print(f"[mesh_viz] no PNG frames under {frames_dir}, skip video.")
		return False

	def _encode_cv2(path_list, out_mp4: str, fps_v: float, frame_hw):
		try:
			import cv2
		except ImportError:
			return False
		h, w = int(frame_hw[0]), int(frame_hw[1])
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(out_mp4, fourcc, float(fps_v), (w, h))
		if not writer.isOpened():
			return False
		for pth in tqdm(path_list, desc="mesh_evolution_video_cv2"):
			bgr = cv2.imread(pth)
			if bgr is None:
				continue
			if bgr.shape[0] != h or bgr.shape[1] != w:
				print(f"[mesh_viz] skip size mismatch: {pth}")
				continue
			writer.write(bgr)
		writer.release()
		return True

	import shutil

	img0 = load_img(paths[0])
	shape = img0.shape[:2]
	print(f"[mesh_viz] encoding {len(paths)} frames → {output_mp4} ({shape[0]}×{shape[1]} @ {fps} fps)")

	ffmpeg_exe = shutil.which("ffmpeg")
	if ffmpeg_exe is None:
		print("[mesh_viz] no 'ffmpeg' on PATH (pip package 'ffmpeg' is not the binary). Using OpenCV mp4v.")
		if _encode_cv2(paths, output_mp4, fps, shape):
			return True
		print("[mesh_viz] OpenCV encode failed; install system ffmpeg: apt install ffmpeg")
		return False

	try:
		video_kwargs = {
			'shape': shape,
			'codec': 'h264',
			'fps': float(fps),
			'crf': 20,
		}
		with media.VideoWriter(output_mp4, **video_kwargs, input_format='rgb') as writer:
			for pth in tqdm(paths, desc="mesh_evolution_video"):
				img = load_img(pth)
				if img.shape[0] != shape[0] or img.shape[1] != shape[1]:
					print(f"[mesh_viz] skip size mismatch: {pth}")
					continue
				img = img / 255.0
				frame_u8 = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
				writer.add_image(frame_u8)
		return True
	except (RuntimeError, OSError) as e:
		print(f"[mesh_viz] mediapy/ffmpeg failed ({e}); trying OpenCV (mp4v).")
		if _encode_cv2(paths, output_mp4, fps, shape):
			return True
		print("[mesh_viz] failed to write MP4; install: apt install ffmpeg")
		return False
