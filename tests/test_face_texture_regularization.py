"""Compare bounded-memory regularization against the original dense objective."""
import unittest
import torch
from utils.face_texture_regularization import accumulate_face_tex_regularization


def dense_loss(tex, tv_weight, l2_weight):
    loss = tex.new_zeros(())
    if tv_weight > 0:
        loss = loss + tv_weight * (
            (tex[:, :, 1:, :] - tex[:, :, :-1, :]).square().mean()
            + (tex[:, 1:, :, :] - tex[:, :-1, :, :]).square().mean())
    if l2_weight > 0:
        loss = loss + l2_weight * tex.square().mean()
    return loss


class RegularizationTests(unittest.TestCase):
    device = "cpu"

    def test_loss_and_gradients_match_dense_with_uneven_chunks(self):
        for dtype in (torch.float32, torch.float64):
            for chunks in (1, 3, 100):
                for weights in ((0.01, 0.01), (0.0, 0.01), (0.01, 0.0)):
                    with self.subTest(dtype=dtype, chunks=chunks, weights=weights):
                        torch.manual_seed(23)
                        reference = torch.randn(7, 8, 8, 3, device=self.device,
                                                dtype=dtype, requires_grad=True)
                        actual = reference.detach().clone().requires_grad_(True)
                        expected_loss = dense_loss(reference, *weights)
                        expected_loss.backward()
                        actual_loss = accumulate_face_tex_regularization(
                            actual, *weights, chunk_faces=chunks)
                        torch.testing.assert_close(actual_loss, expected_loss)
                        torch.testing.assert_close(actual.grad, reference.grad)
                        self.assertFalse(actual_loss.requires_grad)

    def test_adds_to_image_gradients_before_same_adam_update(self):
        torch.manual_seed(67)
        reference = torch.nn.Parameter(torch.randn(11, 4, 4, 3, device=self.device))
        actual = torch.nn.Parameter(reference.detach().clone())
        target = torch.randn_like(reference)
        opt_ref = torch.optim.Adam([reference], lr=0.0005, eps=1e-15)
        opt_actual = torch.optim.Adam([actual], lr=0.0005, eps=1e-15)
        for _ in range(3):
            dense = (reference * target).mean() + dense_loss(reference, 0.01, 0.01)
            dense.backward()
            image_loss = (actual * target).mean()
            image_loss.backward()
            reg = accumulate_face_tex_regularization(actual, 0.01, 0.01, chunk_faces=4)
            torch.testing.assert_close(image_loss.detach() + reg, dense)
            torch.testing.assert_close(actual.grad, reference.grad)
            opt_ref.step()
            opt_actual.step()
            opt_ref.zero_grad(set_to_none=True)
            opt_actual.zero_grad(set_to_none=True)
            torch.testing.assert_close(actual, reference)

    def test_disabled_preserves_existing_gradient(self):
        tex = torch.randn(3, 8, 8, 3, device=self.device, requires_grad=True)
        result = accumulate_face_tex_regularization(tex, 0.0, 0.0)
        self.assertEqual(result.item(), 0.0)
        self.assertIsNone(tex.grad)
        tex.grad = torch.randn_like(tex)
        before = tex.grad.clone()
        accumulate_face_tex_regularization(tex, 0.0, 0.0)
        torch.testing.assert_close(tex.grad, before, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class CudaRegularizationTests(RegularizationTests):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
