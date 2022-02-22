from skfuzzy import control as ctrl     # the control system
import skfuzzy as fuzz                  # fuzzy logic and mf func
import numpy as np                      # arrays and array operations
import matplotlib.pyplot as plt         # Displaying the plots


# >> creating universe of discorse
# New Antecedent/Consequent objects hold universe variables
quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')
rating = ctrl.Consequent(np.arange(0, 11, 1), 'rating')


# >> Creating membership functions
# Auto-membership function population is possible with .automf(3, 5, or 7)
quality.automf(3)
service.automf(3)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 9, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

rating['1 star'] = fuzz.trimf(rating.universe, [0, 0, 3])
rating['2 star'] = fuzz.trimf(rating.universe, [2, 4, 5])
rating['3 star'] = fuzz.trimf(rating.universe, [4, 6, 8])
rating['4 star'] = fuzz.trimf(rating.universe, [7, 8, 10])
rating['5 star'] = fuzz.trimf(rating.universe, [9, 10, 10])

# >> Creating rules
rule1 = ctrl.Rule(quality['poor'] & service['poor'], tip['low'])
rule2 = ctrl.Rule(service['average'], tip['medium'])
rule3 = ctrl.Rule(service['good'] & quality['good'], tip['high'])

rating1 = ctrl.Rule(quality['poor'] & service['poor'], rating['1 star'])
rating2 = ctrl.Rule(quality['poor'] & service['average'], rating['2 star'])
rating3 = ctrl.Rule(quality['poor'] & service['good'], rating['3 star'])
rating4 = ctrl.Rule(quality['average'] & service['poor'], rating['2 star'])
rating5 = ctrl.Rule(quality['average'] & service['average'], rating['3 star'])
rating6 = ctrl.Rule(quality['average'] & service['good'], rating['4 star'])
rating7 = ctrl.Rule(service['good'] & quality['poor'], rating['2 star'])
rating8 = ctrl.Rule(service['good'] & quality['average'], rating['4 star'])
rating9 = ctrl.Rule(service['good'] & quality['good'], rating['5 star'])


# >> Creating the control system
# creating the control system and linking the rules
tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,
                                   rating1, rating2, rating3, rating4, rating5,
                                   rating6, rating7, rating8, rating9])
# used for getting the final results of a system
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
tipping.input['quality'] = 0
tipping.input['service'] = 9

# Crunch the numbers
tipping.compute()

print(tipping.output['rating'])
rating.view(sim=tipping)

plt.show()
