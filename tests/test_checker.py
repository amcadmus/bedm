import numpy as np
from context import bedm
from bedm.tf import TF
from bedm.pd import PD
import unittest

input_1 = np.array([1, 2, 3, 4], dtype = np.float32).reshape([2,2])
input_2 = np.array([4, 3, 2, 1], dtype = np.float32).reshape([2,2])

class TModel(TF.Model):
    def __init__(self, name = None):
        super(TModel, self).__init__(name)
        self.weight \
            = TF.get_variable(
                self,
                'matrix', 
                shape = [2, 2], 
                dtype = TF.float32,
                initializer = TF.constant_initializer(input_1))

    @TF.Model.store_graph
    def forward(self,xx):
        return TF.matmul(self.weight, xx)


class PModel(PD.Model):
    def __init__(self, name = None):
        super(PModel, self).__init__(name)
        self.weight \
            = PD.get_variable(
                self,
                'matrix',
                shape = [2, 2], 
                dtype = PD.float32,
                initializer = PD.constant_initializer(input_1))

    def forward(self,xx):
        return PD.matmul(self.weight, xx)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.ref = np.matmul(input_1, input_2)        

    def test_tf(self):
        tsess = TF.Session()
        tm = TModel('tmodel')
        ti = TF.placeholder(TF.float32, shape = [2,2])             
        tg = tm.forward(ti)
        res = tm.run(tsess, {ti: input_2})[0]

        TF.global_initialize(tsess)
        for ii in range(2):
            for jj in range(2):
                self.assertAlmostEqual(res[ii][jj], self.ref[ii][jj], places = 10)

    def test_pd(self):
        tsess = PD.Session()
        pm = PModel('pmodel')
        pi = PD.placeholder(PD.float32, shape = [2,2])
        tg = pm.forward(pi)
        res = pm.run(tsess, [input_2])

        PD.global_initialize(tsess)        
        for ii in range(2):
            for jj in range(2):
                self.assertAlmostEqual(res[ii][jj], self.ref[ii][jj], places = 10)


if __name__ == "__main__":
    unittest.main()
    
