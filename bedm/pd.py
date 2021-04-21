import numpy as np
import paddle as pd
from .bedm import BEDM

class PD(BEDM):
    float32 = 'float32'
    name = "paddlepaddle"
    class Session():
        def __init__(self, *arg, **kwarg):
            pass
        
        def run(self, tensor_list, feed_dict=None):
            if type(tensor_list) is not list:
                tensor_list = [tensor_list]
            return [ii.numpy() for ii in tensor_list]


    class Model(pd.nn.Layer):
        def __init__(self, *arg, **kwarg):
            if 'name' in kwarg:
                name = kwarg['name']
            else:
                name = None
            super(PD.Model, self).__init__(name)

        def run(self, sess, inputs):
            new_i = [pd.to_tensor(ii) for ii in inputs]
            return self.__call__(*new_i).numpy()


    @staticmethod
    def placeholder(
            dtype,
            shape = None,
            name = None):
        return pd.to_tensor(np.zeros(shape, dtype=np.float32), dtype=dtype)
        # return pd.fluid.layers.data(name=name, shape=shape, dtype=dtype)

    @staticmethod
    def constant_initializer(
            value
    ):
        return pd.nn.initializer.Assign(value)

    @staticmethod
    def initialized_global(        
            sess
    ):
        pass

    @staticmethod
    def get_variable(        
            model,
            name,
            shape = None,
            dtype = None,
            initializer = None
    ):
        return model.create_parameter(
            shape = shape, 
            dtype = dtype,
            default_initializer = initializer)

    @staticmethod
    def matmul(
            aa,
            bb, 
            transpose_a = False, 
            transpose_b = False, 
            name = None
    ):
        return pd.matmul(aa, bb, transpose_x = transpose_a, transpose_y = transpose_b)

