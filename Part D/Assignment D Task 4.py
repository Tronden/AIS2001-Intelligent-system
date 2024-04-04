import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Step 1: Define Fuzzy Variables and Membership Functions
temperature = ctrl.Antecedent(np.arange(0, 51, 1), 'temperature')
temp_error = ctrl.Antecedent(np.arange(-25, 26, 1), 'temp_error')
valve = ctrl.Consequent(np.arange(-100, 101, 1), 'valve')

# Membership functions for temperature
temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 25])
temperature['ideal'] = fuzz.trimf(temperature.universe, [20, 25, 30])
temperature['hot'] = fuzz.trimf(temperature.universe, [25, 50, 50])

# Membership functions for temperature error
temp_error['negative'] = fuzz.zmf(temp_error.universe, -25, 0)
temp_error['zero'] = fuzz.gaussmf(temp_error.universe, 0, 5)
temp_error['positive'] = fuzz.smf(temp_error.universe, 0, 25)

# Membership functions for valve adjustment
valve['close'] = fuzz.zmf(valve.universe, -100, 0)
valve['no_change'] = fuzz.gaussmf(valve.universe, 0, 25)
valve['open'] = fuzz.smf(valve.universe, 0, 100)

# Step 2: Set Up Fuzzy Rules
rule1 = ctrl.Rule(temperature['cold'] | temp_error['negative'], valve['open'])
rule2 = ctrl.Rule(temperature['ideal'], valve['no_change'])
rule3 = ctrl.Rule(temperature['hot'] | temp_error['positive'], valve['close'])

# Step 3: Create and Simulate Fuzzy Control System
temperature_control = ctrl.ControlSystem([rule1, rule2, rule3])
temperature_controller = ctrl.ControlSystemSimulation(temperature_control)

# Example: Adjusting to a Desired Temperature
desired_temperature = 25  # Desired temperature (degrees Celsius)
current_temperature = 20  # Current temperature (degrees Celsius)

temperature_controller.input['temperature'] = current_temperature
temperature_controller.input['temp_error'] = desired_temperature - current_temperature
temperature_controller.compute()

valve_position = temperature_controller.output['valve']
print(f"Valve Adjustment: {valve_position:.2f}")
# Depending on the valve position, you'd adjust the water flow to increase, decrease, or maintain the temperature.