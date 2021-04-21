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
    def error(aa, bb, transpose_a=False, transpose_b=False):
        raise NotImplementedError



# a = tf.reshape(tf.constant([1, 2, 3, 4], dtype = tf.float32), [2,2])
# b = tf.reshape(tf.constant([4, 3, 2, 1], dtype = tf.float32), [2,2])
# c = TF.matmul(a,b)

# tsess = TF.Session()
# print(tsess.run(c))

# e = pd.reshape(pd.to_tensor(np.array([1, 2, 3, 4], dtype = np.float32)), [2,2])
# f = pd.reshape(pd.to_tensor(np.array([4, 3, 2, 1], dtype = np.float32)), [2,2])
# g = PD.matmul(e,f)

# # print(g.numpy())
# psess = PD.Session()
# print(psess.run(g))


# a = TF.placeholder(TF.float32, shape = [2,2])
# b = TF.placeholder(TF.float32, shape = [2,2])
# c = TF.matmul(a,b)
# fd = {a:input_1, b:input_2}
# print(tsess.run(c, feed_dict=fd))




# tm = TModel('tmodel')
# pm = PModel('pmodel')

# ti = TF.placeholder(TF.float32, shape = [2,2])
# pi = PD.placeholder(PD.float32, shape = [2,2])

# init_op = tf.global_variables_initializer()
# tsess.run(init_op)

# tg = tm.forward(ti)
# pg = pm.forward(pi)

# print(tm.run(tsess, {ti: input_2}))
# print(pm.run(psess, [input_2]))

# # print(pm(pd.to_tensor(input_2)))

