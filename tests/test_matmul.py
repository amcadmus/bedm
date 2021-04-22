import numpy as np
from context import bedm
from bedm.tf import TF
from bedm.pd import PD
import unittest

class TestMatMul(unittest.TestCase):
    def setUp(self):
        self.m = 3
        self.n = 4
        self.k = 2
        self.mat_0 = np.arange(0, self.m*self.k, dtype=float).reshape([self.m, self.k])
        self.mat_0_t = self.mat_0.T
        self.mat_1 = np.arange(self.k*self.n, 0, -1, dtype=float).reshape([self.k, self.n])
        self.mat_1_t = self.mat_1.T
        # use numpy matmul as reference
        self.ref = np.matmul(self.mat_0, self.mat_1)
        
    def beckend_test(self, be):
        mat_0 = be.constant(self.mat_0)
        mat_1 = be.constant(self.mat_1)
        prod = be.matmul(mat_0, mat_1)
        sess = be.Session()
        res = sess.run(prod)[0]
        for ii in range(self.m):
            for jj in range(self.n):
                self.assertAlmostEqual(res[ii][jj], self.ref[ii][jj])

    def beckend_test_trans(self, be):
        mat_0 = be.constant(self.mat_0_t)
        mat_1 = be.constant(self.mat_1_t)
        prod = be.matmul(mat_0, mat_1, transpose_a=True, transpose_b=True)
        sess = be.Session()
        res = sess.run(prod)[0]
        for ii in range(self.m):
            for jj in range(self.n):
                self.assertAlmostEqual(res[ii][jj], self.ref[ii][jj])

    def test_tf(self):
        self.beckend_test(TF)

    def test_pd(self):
        self.beckend_test(PD)

    def test_trans_tf(self):
        self.beckend_test(TF)

    def test_trans_pd(self):
        self.beckend_test(PD)
