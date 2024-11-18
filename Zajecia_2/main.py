import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Creating input variables (Antecedent) for cleanliness, location, and customer service
cleanliness = ctrl.Antecedent(np.arange(0, 11, 1), 'cleanliness')
location = ctrl.Antecedent(np.arange(0, 11, 1), 'location')
customer_service = ctrl.Antecedent(np.arange(0, 11, 1), 'customer_service')

# Creating the output variable (Consequent) for hotel quality
hotel_quality = ctrl.Consequent(np.arange(0, 101, 1), 'hotel_quality')

# Automatically creating membership functions for inputs
cleanliness.automf(3)
location.automf(3)
customer_service.automf(3)

# Creating membership functions for the output
hotel_quality['low'] = fuzz.trimf(hotel_quality.universe, [0, 0, 50])
hotel_quality['medium'] = fuzz.trimf(hotel_quality.universe, [0, 50, 100])
hotel_quality['high'] = fuzz.trimf(hotel_quality.universe, [50, 100, 100])

# View membership functions
cleanliness['average'].view()
location.view()
customer_service.view()
hotel_quality.view()

# Defining fuzzy rules
# Rule 1: low
rule1 = ctrl.Rule(cleanliness['poor'] | location['poor'] | customer_service['poor'], hotel_quality['low'])

# Rule 2: medium
rule2 = ctrl.Rule(cleanliness['average'] & location['average'] & customer_service['average'], hotel_quality['medium'])

# Rule 3: high
rule3 = ctrl.Rule(cleanliness['good'] & location['good'] & customer_service['good'], hotel_quality['high'])

# Creating the hotel quality control system defined by the rules
hotel_quality_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

# Creating a simulation for the control system
hotel_quality_simulation = ctrl.ControlSystemSimulation(hotel_quality_ctrl)

def evaluate_hotel_quality(cleanliness_score, location_score, customer_service_score):
    """
    Function to evaluate hotel quality based on scores for cleanliness, location, and customer service.

    :param cleanliness_score: Cleanliness score (from 0 to 10)
    :param location_score: Location score (from 0 to 10)
    :param customer_service_score: Customer service score (from 0 to 10)
    """
    # Setting input values for the system simulation
    hotel_quality_simulation.input['cleanliness'] = cleanliness_score
    hotel_quality_simulation.input['location'] = location_score
    hotel_quality_simulation.input['customer_service'] = customer_service_score

    # Performing calculations in the fuzzy system
    hotel_quality_simulation.compute()

    # Displaying the hotel quality evaluation result
    print("Hotel quality evaluation:", hotel_quality_simulation.output['hotel_quality'])
    hotel_quality.view(sim=hotel_quality_simulation)

# Example evaluations for three cases
evaluate_hotel_quality(3, 4, 5)
evaluate_hotel_quality(5, 6, 7)
evaluate_hotel_quality(10, 9, 10)

# Display all plots
plt.show()
