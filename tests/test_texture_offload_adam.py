"""Numerical checks against ordinary Adam, including live state and SH groups."""
import unittest
import torch
from utils.texture_offload_adam import TextureOffloadAdam

class OffloadTests(unittest.TestCase):
    device = 'cpu'

    def exercise(self, warm):
        torch.manual_seed(172)
        a = torch.nn.Parameter(torch.randn(11,16,16,3,device=self.device)*.1)
        b = torch.nn.Parameter(a.detach().clone())
        x = torch.nn.Parameter(torch.randn(7,3,device=self.device))
        y = torch.nn.Parameter(x.detach().clone())
        ref = torch.optim.Adam([{'params':[a],'name':'face_textures'}, {'params':[x],'name':'f_dc','lr':.004}],lr=.0005,eps=1e-15)
        opt = torch.optim.Adam([{'params':[b],'name':'face_textures'}, {'params':[y],'name':'f_dc','lr':.004}],lr=.0005,eps=1e-15)
        for step in range(6):
            if step == warm:
                opt = TextureOffloadAdam.from_adam(opt,chunk_faces=4)
            # Also exercise steps with no texture gradient and LR changes.
            g = torch.randn_like(a) if step != 4 else None
            gx = torch.randn_like(x)
            a.grad = g; b.grad = g.clone() if g is not None else None
            x.grad = gx; y.grad = gx.clone()
            ref.param_groups[0]['lr'] = opt.param_groups[0]['lr'] = .0005/(step+1)
            ref.step(); opt.step()
            for left,right in [(a,b),(x,y)]:
                torch.testing.assert_close(left,right,rtol=2e-5,atol=2e-7)
                for key in ['step','exp_avg','exp_avg_sq']:
                    torch.testing.assert_close(ref.state[left][key].cpu(),opt.state[right][key].cpu(),rtol=2e-5,atol=2e-7)
            if step >= warm:
                self.assertEqual(opt.state[b]['exp_avg'].device.type,'cpu')
                if self.device == 'cuda': self.assertTrue(opt.state[b]['exp_avg'].is_pinned())
            ref.zero_grad(set_to_none=True); opt.zero_grad(set_to_none=True)

    def test_cold_start_and_unaligned_chunks(self): self.exercise(warm=0)
    def test_existing_adam_moments_are_preserved(self): self.exercise(warm=2)

    def test_invalid_options_reject_before_updating_other_groups(self):
        x=torch.nn.Parameter(torch.ones(2,2,2,3,device=self.device))
        y=torch.nn.Parameter(torch.ones(3,device=self.device))
        opt=TextureOffloadAdam([{'params':[x],'name':'face_textures','amsgrad':True},{'params':[y],'name':'f_dc'}])
        x.grad=torch.ones_like(x);y.grad=torch.ones_like(y)
        with self.assertRaises(ValueError):opt.step()
        torch.testing.assert_close(y,torch.ones_like(y))
        self.assertEqual(len(opt.state),0)

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class CudaOffloadTests(OffloadTests):
    device='cuda'

if __name__=='__main__':unittest.main()
