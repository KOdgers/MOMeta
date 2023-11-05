import unittest
from mometa import  MoMeta,  reparameterize_model, condense_model
import copy
import torch

class Test(unittest.TestCase):
    def setUp(self) -> None:
        pass
        self.s0 = {"width_multipliers": (0.75, 1.0, 1.0, 2.0),
           "num_conv_branches": 4}
        
    
    def test_residual_post_condense(self):
        bblock = MoMeta(num_classes=1000, inference_mode=False,
                        **self.s0)   
        bblock.eval().cuda() 
        rblock = condense_model(bblock)
        
        rblock.eval().cuda()
        
        condensed_residuals = []
        for _ in range(5):
            x = torch.randn(1, 3, 224, 224).cuda()
            y_base = bblock(x)
            y_new = rblock(x)
            residual = y_base - y_new
            residual = torch.pow(residual, 2).sum()
            condensed_residuals.append(residual)
            
        self.assertLess(condensed_residuals, [0.1]*len(condensed_residuals))
        



if __name__ == '__main__':
    unittest.main() 