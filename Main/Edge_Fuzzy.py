from skfuzzy import control as ctrl     # the control system simulaion
import skfuzzy as fuzz                  # fuzzy logic and mf
import numpy as np                      # arrays and array operations


# >> dPj fuzzy set
# This creates the UOD and membership function
# for a single pixil value
def create_bit_fuzzy_set(id):
    # creating universe of discorse (0~255)
    uod_contrast = ctrl.Antecedent(np.arange(0, 256, 1), 'contrast'+id)
    uod_contrast["lower"] = fuzz.trapmf(        # creating and inserting the "lower" set
        uod_contrast.universe,  [10, 10, 75, 200])
    uod_contrast["higher"] = fuzz.trapmf(       # creating and inserting the "Higher" set
        uod_contrast.universe, [30, 75, 255, 255])
    return uod_contrast
    # note that [5,5,150,200] and [200,255,255,255] can be used for more complex images
    # like the pepper image provided, or a face.
    # original: [5,5,25,75] and [25,75,255,255]


# >> output fuzzy set
# This creates the UOD and membership function
# for the final edge decision
def create_edge_fuzzy_set():
    # creating universe of discorse (0~255)
    uod_edge = ctrl.Consequent(np.arange(0, 256, 1), 'edge')
    # creating and inserting the "non-edge" set
    uod_edge["non-edge"] = fuzz.gaussmf(uod_edge.universe, 8.5, 1.0)
    # create and inserting the "edge" set
    uod_edge["edge"] = fuzz.gaussmf(uod_edge.universe, 240.0, 1.0)
    return uod_edge


# >> creating a size 8 array of pixel membership functions
# returns list of size 8
def create_matrixOf_bit_fuzzy_set():
    return_set_mat = []     # declaring the matrix
    for i in range(8):      # looping the creation of the pixel membership function
        return_set_mat.append(create_bit_fuzzy_set(str(i)))
    return return_set_mat


# >> creating the control simulation
# creating the rule base, and combining the Consequent and Antecedents
def create_control_sim(set_array, output_set):
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

    return ctrl.ControlSystemSimulation(bit_control)


# >> returns if edge or not
# takes in the simulation with the appropriate inputs
# then returns a true if it is an edge, false otherwise
def isEdge(simulation):
    # checking if the simulation computes
    try:
        # if it does then we have an edge
        simulation.compute()
        return True
    except:
        # otherwise non-edge
        return False


# contrast adjustment
# =======================================================
# =======================================================
def _create_intensity_fuzzy_set():
    # creating universe of discorse (0~255)
    uod_intensity = ctrl.Antecedent(np.arange(0, 256, 1), 'intensity')
    uod_intensity["Darker"] = fuzz.trapmf(
        uod_intensity.universe, [0, 0, 80, 150])
    uod_intensity["Gray"] = fuzz.trimf(uod_intensity.universe, [20, 80, 140])
    uod_intensity["Brighter"] = fuzz.trapmf(
        uod_intensity.universe, [80, 150, 255, 255])
    return uod_intensity


def _create_output_intensity_fuzzy_set():
    # creating universe of discorse (0~255)
    uod_intensity_o = ctrl.Consequent(np.arange(0, 256, 1), 'output')
    # creating and inserting the "non-edge" set
    uod_intensity_o["Darkest"] = fuzz.gaussmf(
        uod_intensity_o.universe, 23, 4)
    uod_intensity_o["Gray"] = fuzz.gaussmf(
        uod_intensity_o.universe, 128, 4)
    uod_intensity_o["Brightest"] = fuzz.gaussmf(
        uod_intensity_o.universe, 225, 4)
    return uod_intensity_o


def create_intensity_sim():
    intensity = _create_intensity_fuzzy_set()
    output = _create_output_intensity_fuzzy_set()
    rule1 = ctrl.Rule(intensity['Darker'], output['Darkest'])
    rule2 = ctrl.Rule(intensity['Gray'], output['Gray'])
    rule3 = ctrl.Rule(intensity['Brighter'], output['Brightest'])
    intensity_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    return ctrl.ControlSystemSimulation(intensity_ctrl)
