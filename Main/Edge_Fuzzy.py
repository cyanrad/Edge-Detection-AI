from skfuzzy import control as ctrl     # the control system
import skfuzzy as fuzz                  # fuzzy logic and mf func
import numpy as np                      # arrays and array operations
import matplotlib.pyplot as plt         # Displaying the plots


# >> dPj fuzzy set
# This creates the UOD and membership function
# for a single pixil value
def create_bit_fuzzy_set(id):
    uod_contrast = ctrl.Antecedent(np.arange(0, 256, 1), 'contrast'+id)
    uod_contrast["lower"] = fuzz.trapmf(uod_contrast.universe, [0, 0, 25, 75])
    uod_contrast["higher"] = fuzz.trapmf(
        uod_contrast.universe, [25, 75, 255, 255])
    return uod_contrast


def create_edge_fuzzy_set():
    uod_edge = ctrl.Consequent(np.arange(0, 256, 1), 'edge')
    uod_edge["non-edge"] = fuzz.gaussmf(uod_edge.universe, 8.5, 1.0)
    uod_edge["edge"] = fuzz.gaussmf(uod_edge.universe, 240.0, 1.0)
    return uod_edge


def create_matrixOf_bit_fuzzy_set():
    return_set_mat = []
    for i in range(8):
        return_set_mat.append(create_bit_fuzzy_set(str(i)))
    return return_set_mat


def create_rule_base(set_array, output_set):
    bit_control = ctrl.ControlSystem()

    # generating edge rules
    higher_1 = np.array([0, 0, 1, 3])
    higher_2 = np.array([1, 3, 2, 5])
    lower = np.array([4, 6, 7])
    for i in range(3):
        for j in range(4):
            temp_rule = ctrl.Rule(
                set_array[higher_1[j]]['higher'] &
                set_array[higher_2[j]]['higher'] &
                set_array[lower[i]]['lower'],
                output_set['edge'])
            bit_control.addrule(temp_rule)

    # generating non-edge rules

    return ctrl.ControlSystemSimulation(bit_control)
