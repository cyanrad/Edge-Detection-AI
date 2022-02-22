from skfuzzy import control as ctrl     # the control system
import skfuzzy as fuzz                  # fuzzy logic and mf func
import numpy as np                      # arrays and array operations
import matplotlib.pyplot as plt         # Displaying the plots


# >> dPj fuzzy set
# This creates the UOD and membership function
# for a single pixil value
def create_bit_fuzzy_set():
    uod_contrast = ctrl.Antecedent(np.arange(0, 256, 1), 'contrast')
    uod_contrast["lower"] = fuzz.trapmf(uod_contrast.universe, [0, 0, 25, 75])
    uod_contrast["higher"] = fuzz.trapmf(
        uod_contrast.universe, [25, 75, 255, 255])
    return uod_contrast


def create_matrixOf_bit_fuzzy_set():
    return_set_mat = []
    for i in range(8):
        return_set_mat.append(create_bit_fuzzy_set())
    return return_set_mat
