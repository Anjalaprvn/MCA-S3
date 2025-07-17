#   c=np.array([
    [2,4],
    [3,5]
])
c
OUTPUT:
array([[2, 4],
       [3, 5]])

#   d=np.array([
    [4,8],
    [6,9]
])
d

OUTPUT:
array([[4, 8],
       [6, 9]])

#   c+d

OUTPUT:
array([[ 6, 12],
       [ 9, 14]])

#  e=c*d
    e

OUTPUT:
array([[ 8, 32],
       [18, 45]])

#  f=np.dot(c,d)
f

OUTPUT:
array([[32, 52],
       [42, 69]])

#  c.T
OUTPUT:
array([[2, 3],
       [4, 5]])
#  f.T
OUTPUT:
array([[32, 42],
       [52, 69]])
#print("\n matrix 1:\n",c)
print("\n matrix 2:\n",d)
print("\n addition:\n",c+d)
print("\n substraction\n",c-d)
print("\n multiplication(element wise):\n",c*d)
print("\n dot multiplication (row x coloumn):\n",f)
print("\n Transpose:\n",f.T)

OUTPUT:

 matrix 1:
 [[2 4]
 [3 5]]

 matrix 2:
 [[4 8]
 [6 9]]

 addition:
 [[ 6 12]
 [ 9 14]]

 substraction
 [[-2 -4]
 [-3 -4]]

 multiplication(element wise):
 [[ 8 32]
 [18 45]]

 dot multiplication (row x coloumn):
 [[32 52]
 [42 69]]

 Transpose:
 [[32 42]
 [52 69]]




