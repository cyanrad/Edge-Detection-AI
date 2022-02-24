import Edge_Fuzzy as ef
import pixil_manip as pm
import numpy as np
import matplotlib.pyplot as plt         # Displaying the plots


testing_arr = ef.create_matrixOf_bit_fuzzy_set()
testing_edge = ef.create_edge_fuzzy_set()
temp = ef.create_rule_base(testing_arr, testing_edge)

testing_mat = np.matrix([[10, 10, 197], [199, 199, 198], [197, 50, 199]])
testing_mat_2 = pm.get_dPj_matrix(testing_mat)

pm.apply_fuzzy_contrast(temp, testing_mat_2)

# very shitty terrible code, science, and logic
# god i wish i were a duck
try:
    temp.compute()
    print(temp.output["edge"])
    testing_edge.view(sim=temp)
except ValueError:
    print("not an edge")


plt.show()
