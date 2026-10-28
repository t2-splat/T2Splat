# Project adaptations maintained by zhtang2004@gmail.com.
"""Dense Adam with CPU-resident texture moments and bounded GPU update buffers.

Only the group named face_textures is offloaded. Parameters/gradients stay on
GPU, and all other groups use ordinary torch.optim.Adam. No quantization or
sparse update approximation is used. CPU transfers trade speed for GPU memory.
"""
import math
import torch


class TextureOffloadAdam(torch.optim.Adam):
    def __init__(self, params, chunk_faces=65536, **kwargs):
        if chunk_faces <= 0:
            raise ValueError("chunk_faces must be positive")
        self.chunk_faces = int(chunk_faces)
        super().__init__(params, **kwargs)

    @classmethod
    def from_adam(cls, optimizer, chunk_faces=65536):
        if type(optimizer) is not torch.optim.Adam:
            raise TypeError("Expected ordinary torch.optim.Adam")
        result = cls(optimizer.param_groups, chunk_faces=chunk_faces, **optimizer.defaults)
        result.state = optimizer.state
        return result

    @staticmethod
    def _validate(group, param, gradient):
        if param.ndim != 4 or param.dtype not in (torch.float32, torch.float64):
            raise ValueError("Offloaded textures must be dense floating point [F,R,R,C] tensors")
        if gradient.is_sparse or not param.is_contiguous() or not gradient.is_contiguous():
            raise ValueError("Offloaded textures require contiguous dense parameters and gradients")
        for flag in ("amsgrad", "maximize", "capturable", "differentiable", "fused", "decoupled_weight_decay"):
            if group.get(flag, False):
                raise ValueError(f"Offloaded texture Adam does not support {flag}")
        if group.get("weight_decay", 0) != 0:
            raise ValueError("Use the existing texture L2 loss, not optimizer weight decay")
        if isinstance(group['lr'], torch.Tensor) or any(isinstance(b, torch.Tensor) for b in group['betas']):
            raise ValueError("Offloaded texture Adam requires scalar learning rate and betas")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("TextureOffloadAdam does not support closures")
        pending = [(group, p, p.grad) for group in self.param_groups
                   if group.get('name') == 'face_textures'
                   for p in group['params'] if p.grad is not None]
        for group, p, gradient in pending:
            self._validate(group, p, gradient)
        # Avoid ordinary Adam allocating full texture moments on the GPU.
        for _, p, _ in pending:
            p.grad = None
        try:
            result = super().step()
        finally:
            for _, p, gradient in pending:
                p.grad = gradient
        for group, param, gradient in pending:
            state = self.state[param]
            if 'step' not in state:
                state['step'] = torch.tensor(0., dtype=torch.float32)
            elif state['step'].device.type != 'cpu':
                state['step'] = state['step'].cpu()
            for name in ('exp_avg', 'exp_avg_sq'):
                previous = state.get(name)
                if previous is None or previous.device.type != 'cpu' or (param.is_cuda and not previous.is_pinned()):
                    host = torch.empty(param.shape, dtype=param.dtype, device='cpu', pin_memory=param.is_cuda)
                    if previous is None:
                        host.zero_()
                    else:
                        host.copy_(previous)
                    state[name] = host
            step = int(state['step'].item()) + 1
            beta1, beta2 = group['betas']
            step_size = group['lr'] / (1 - beta1 ** step)
            correction2 = math.sqrt(1 - beta2 ** step)
            for start in range(0, len(param), self.chunk_faces):
                end = min(start + self.chunk_faces, len(param))
                grad = gradient[start:end]
                m = state['exp_avg'][start:end].to(param.device)
                v = state['exp_avg_sq'][start:end].to(param.device)
                m.lerp_(grad, 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = v.sqrt().div_(correction2).add_(group['eps'])
                param[start:end].addcdiv_(m, denom, value=-step_size)
                # Blocking copies keep host moments valid before the next chunk.
                state['exp_avg'][start:end].copy_(m)
                state['exp_avg_sq'][start:end].copy_(v)
                del m, v, denom
            state['step'].fill_(step)
        return result
