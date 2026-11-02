"""Check that DTU foreground priors cannot supervise background geometry."""
import unittest
import torch
from utils.loss_utils import inverse_depth_prior_loss, normal_prior_loss


class PriorLossTests(unittest.TestCase):
    def test_depth_ignores_background_nonfinite_and_nonpositive_targets(self):
        pred = torch.tensor([[[2., 100., 3., 4., 5.]]], requires_grad=True)
        target = torch.tensor([[[1., 1., float('nan'), 0., -1.]]])
        loss = inverse_depth_prior_loss(pred, target, torch.tensor([[1., 0., 1., 1., 1.]]))
        self.assertEqual(loss.item(), 1.)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([[[1., 0., 0., 0., 0.]]]))

    def test_depth_is_normalized_by_valid_foreground_coverage(self):
        pred = torch.tensor([[[2., 5., 999.]]], requires_grad=True)
        target = torch.ones_like(pred)
        loss = inverse_depth_prior_loss(pred, target, torch.tensor([[[1., .5, 0.]]]))
        self.assertEqual(loss.item(), 2.)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([[[2/3, 1/3, 0.]]]))

    def test_normals_ignore_background_invalid_and_zero_vectors(self):
        pred = torch.tensor([[[0., 1., 1., 1.]], [[0., 0., 0., 0.]], [[1., 0., 0., 0.]]], requires_grad=True)
        target = torch.tensor([[[0., 1., float('nan'), 0.]], [[0., 0., 0., 0.]], [[2., 0., 0., 0.]]])
        loss = normal_prior_loss(pred, target, torch.tensor([[[1., 0., 1., 1.]]]))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        expected = torch.zeros_like(pred);expected[2,0,0] = -1
        torch.testing.assert_close(pred.grad, expected)

    def test_empty_prior_mask_returns_differentiable_zero(self):
        for fn in (inverse_depth_prior_loss, normal_prior_loss):
            with self.subTest(function=fn.__name__):
                pred = torch.ones((3 if fn is normal_prior_loss else 1, 2, 2), requires_grad=True)
                loss = fn(pred, torch.ones_like(pred), torch.zeros((1,2,2)))
                self.assertEqual(loss.item(), 0.)
                loss.backward()
                self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))


if __name__ == '__main__':unittest.main()
