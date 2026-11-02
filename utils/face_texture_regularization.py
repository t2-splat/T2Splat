# Project adaptations maintained by zhtang2004@gmail.com.
"""Bounded-memory TV/L2 gradient accumulation for per-face textures."""
import torch


def accumulate_face_tex_regularization(face_tex, lambda_tv, lambda_l2,
                                      chunk_faces=65536):
    """Add exact first-order regularization gradients to ``face_tex.grad``.

    Call after the image loss backward pass and before mirror-gradient
    redirection / optimizer.step(). Faces do not share TV edges, so chunks
    can be differentiated independently, weighted by their face counts.
    Detached chunk leaves avoid allocating a full-sized SliceBackward
    gradient for every chunk. The returned loss is detached, for logging.
    """
    if chunk_faces <= 0:
        raise ValueError("chunk_faces must be positive")
    total = face_tex.new_zeros(())
    n_faces = face_tex.shape[0]
    if n_faces == 0 or (lambda_tv <= 0 and lambda_l2 <= 0):
        return total
    if face_tex.grad is None:
        face_tex.grad = torch.zeros_like(face_tex)

    for start in range(0, n_faces, chunk_faces):
        end = min(start + chunk_faces, n_faces)
        with torch.enable_grad():
            chunk = face_tex[start:end].detach().requires_grad_(True)
            loss = chunk.new_zeros(())
            if lambda_tv > 0:
                tv_x = (chunk[:, :, 1:, :] - chunk[:, :, :-1, :]).square().mean()
                tv_y = (chunk[:, 1:, :, :] - chunk[:, :-1, :, :]).square().mean()
                loss = loss + lambda_tv * (tv_x + tv_y)
            if lambda_l2 > 0:
                loss = loss + lambda_l2 * chunk.square().mean()
            loss = loss * ((end - start) / n_faces)
            gradient, = torch.autograd.grad(loss, chunk)
        with torch.no_grad():
            face_tex.grad[start:end].add_(gradient)
            total.add_(loss.detach())
        del chunk, loss, gradient
    return total
