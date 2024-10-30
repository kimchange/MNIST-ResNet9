import numpy as np
np.random.seed(0)
a = np.random.rand(2,2,3,4).reshape(2,2,3,4)

val = np.max(a.reshape(a.shape[0],a.shape[1],-1), axis=-1)
index = np.argmax(a.reshape(a.shape[0],a.shape[1],-1),axis=-1,keepdims=True)

val2 = np.take_along_axis( a.reshape(a.shape[0],a.shape[1],-1), index[:,:],axis=-1)

np.put_along_axis( a.reshape(a.shape[0],a.shape[1],-1), index[:,:], val2, axis=-1)

val3 = [a.reshape(a.shape[0],a.shape[1],-1)[bb,cc,index[bb,cc]] for bb in range(a.shape[0]) for cc in range(a.shape[1])]

val3 = np.stack(val3,axis=-1).reshape(a.shape[0],a.shape[1])

