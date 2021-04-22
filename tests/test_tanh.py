import numpy as np
from context import bedm
from bedm.tf import TF
from bedm.pd import PD
import unittest

class TestTanh(unittest.TestCase):
    def setUp(self):
        self.m = 3
        self.k = 2
        self.mat_0 = np.arange(0, self.m*self.k, dtype=float).reshape([self.m, self.k])
        # use numpy matmul as reference
        self.ref = np.tanh(self.mat_0)
        
    def beckend_test(self, be):
        mat_0 = be.constant(self.mat_0)
        tanh = be.tanh(mat_0)
        sess = be.Session()
        res = sess.run(tanh)[0]
        for ii in range(self.m):
            for jj in range(self.k):
                self.assertAlmostEqual(res[ii][jj], self.ref[ii][jj])

    def test_tf(self):
        self.beckend_test(TF)

    def test_pd(self):
        self.beckend_test(PD)
