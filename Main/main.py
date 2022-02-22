import Edge_Fuzzy as ef
import pixil_manip as pm
import numpy as np

#testing = ef.create_matrixOf_bit_fuzzy_set()
testing_mat = np.matrix([[100, 150, 200], [100, 120, 140], [210, 130, 90]])
print(testing_mat)
testing_mat_2 = pm.get_dPj_matrix(testing_mat)
print(testing_mat_2)
