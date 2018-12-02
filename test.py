import numpy as np
# word_dic = {'a':1, 'b':2, 'c':3}

# print(word_dic, type(word_dic))
# print(word_dic.items(), type(word_dic.items()))

# word_dic.items()
# print(word_dic.get(1, '?'))

# tuple_list = [('a', 1), ('b', 2), ('c', 3)]
k = np.array(['a', 'b', 'c'])
v = np.array([1, 2, 3])
list_tuple =(k, v)
k,v = list_tuple

print(list_tuple, k, v)

order = np.argsort(np.random.random(v.shape))
print(order)
k = k[[0]]
v = v[order]
print(k,v)