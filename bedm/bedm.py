from abc import ABC, abstractmethod
import paddle as pd

class BEDM(ABC):
    @staticmethod
    def matmul(
            aa,
            bb, 
            transpose_a = False, 
            transpose_b = False, 
            name = None
    ):
        raise NotImplementedError

    @staticmethod
    def placeholder(
            dtype,
            shape = None,
            name = None):
        raise NotImplementedError

    @staticmethod
    def constant_initializer(
            value
    ):
        raise NotImplementedError

    @staticmethod
    def global_initialize(        
            sess
    ):
        raise NotImplementedError
    
    @staticmethod
    def get_variable(        
            model,
            name,
            shape = None,
            dtype = None,
            initializer = None
    ):
        raise NotImplementedError
    
    @staticmethod
    def matmul(
            aa,
            bb, 
            transpose_a = False, 
            transpose_b = False, 
            name = None
    ):
        raise NotImplementedError

    @staticmethod
    def tanh(
            xx,
            name = None
    ):
        raise NotImplementedError

    
