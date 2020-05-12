import unittest

import torch

from modules import OurModule


class ModulesTest(unittest.TestCase):
    def test_main(self):
        net = OurModule(num_inputs=2, num_classes=3)
        print(net)
        v = torch.FloatTensor([[2, 3]])
        out = net(v)
        print(out)
        print("Cuda's availability is %s" % torch.cuda.is_available())
        if torch.cuda.is_available():
            print("Data from cuda: %s" % out.to('cuda'))


if __name__ == '__main__':
    unittest.main()
