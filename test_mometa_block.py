import unittest
from mobile_blocks import MOMetaBlock
import copy
import torch

class Test(unittest.TestCase):
    def setUp(self) -> None:
        pass
    
    def test_residual_post_condense(self):
         
        bblock = MOMetaBlock(in_channels = 3,
                                out_channels = 32,
                                kernel_size = 3,
                                stride = 2,
                                padding = 1)
        rblock = copy.deepcopy(bblock)
        rblock.condense()
        
        bblock.eval().cuda()
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