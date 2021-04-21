import numpy as np
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf
from .bedm import BEDM

class TF(BEDM):
    float32 = tf.float32
    name = "TensorFlow"
    class Session():
        def __init__(self, *arg, **kwarg):
            self.sess = tf.Session(*arg, **kwarg)

        def run(self, tensor_list, feed_dict=None):
            if type(tensor_list) is not list:
                tensor_list = [tensor_list]
            return self.sess.run(tensor_list, feed_dict=feed_dict)

    class Model():
        def __init__(self, *arg, **kwarg):
            pass
    
        def forward(self, *arg, **kwarg):
            pass

        def run(self, sess, inputs):
            return sess.run(self.graph, feed_dict=inputs)

        def store_graph(func):
            def wrapper(self, xx):
                self.graph = func(self, xx)
                return self.graph
            return wrapper

    @staticmethod
    def placeholder(
            dtype,
            shape = None,
            name = None):
        return tf.placeholder(dtype, shape=shape, name=name)

    @staticmethod
    def constant_initializer(
            value
    ):
        return tf.constant_initializer(value)


    @staticmethod
    def initialized_global(        
            sess
    ):
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

    @staticmethod
    def get_variable(        
            model,
            name,
            shape = None,
            dtype = None,
            initializer = None
    ):
        return tf.get_variable(
            name, 
            shape = shape, 
            dtype = dtype,
            initializer = initializer)

    @staticmethod
    def matmul(
            aa,
            bb, 
            transpose_a = False, 
            transpose_b = False, 
            name = None
    ):
        return tf.matmul(aa, bb, transpose_a=transpose_a, transpose_b=transpose_b, name=name)

