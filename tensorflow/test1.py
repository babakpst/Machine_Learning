
import tensorflow as tf

#=====
print(tf.__version__)

print()
print()
x = tf.constant(3)
print(x)

y = tf.Variable(9)
print(y)

y =12
print(y)

x = 8
print(x)


x1 = tf.constant([1,2,3])
print(x1)


print("one var: ", x1[1])

x2 = tf.constant([[1],[2],[3]])
print(x2)


xx = tf.constant([
                  [1,2,3],
                  [4,5,6]
                 ])

yy = tf.Variable([[1,2,3],[4,5,6]])
print(xx)
print(yy)



print(" variable tensor: ", yy[1,1])


xxx = tf.stack([xx,xx])
print()
print(xxx)

yy = tf.stack([yy,yy])
print()
print(yy)

xx2 = xx[:,1]
print(xx2.shape)
print(xx2)

#============
# python read numbers row-by-row, so when we reshape tensors, it put them in the new shape row-wise.

xr = tf.reshape(xx,[3,2])
print(xr)

#====== reassign/reshape
print("  ========== var ")
nv = tf.Variable(3, dtype=tf.float32, name='var') # their values can change, there are three methods to change the value.
nc = tf.constant(5, dtype=tf.float32, name='const') # costant variables, there is no method to change the values. 

print("var: ", nv)
nv.assign(12.5)
print("var: ", nv)
nv.assign_add(12.5)
print("var: ", nv)
nv.assign_sub(12.5)
print("var: ", nv)


# this is a constant var, and it does not have assign attribute.
#print("cons: ", nc)
#nc.assign(12.5)
#print("const: ", nc)





