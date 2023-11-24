#README - You are required to read and agree to the following statements before using this code.

#Description:
#Plotting & calculation code for CHE260 Thermodynamics & Heat Transfer, Lab 2 - First Law of Thermodynamics

#Disclaimer:
#This thermodynamics lab code was exclusively developed by Aspen Erlandsson, an Engineering Science student at
#the University of Toronto at time of writing[2023-11-23], as part of an academic project for the course CHE260.

#The code is made available for educational and informational purposes only. While I have made every
#effort to ensure the accuracy and effectiveness of this code, it is provided "as is" without any warranty
#of any kind, either expressed or implied.

#Additionally, any use of this code, in whole or in part, in academic or commercial works must include
#proper citation to avoid academic misconduct and copyright infringement. This code is intended for replication
#and review of results presented in the report, not for use by other students completing this lab. Please use
#this code responsibly, respecting academic integrity and the intellectual effort put into its creation.




import matplotlib.pyplot as plt
import numpy as np

# Initialize dictionaries to store data for multiple files
data_dict = {
    "pt_1_t_a": {
        "time": [],
        "T1": [],
        "P1": [],
        "mass_flowrate_gmin": [],
        "heater_energy_kj": [],
    },

    "pt_2_t_a": {
        "time": [],
        "T1": [],
        "P1": [],
        "mass_flowrate_gmin": [],
        "heater_energy_kj": [],
    },

    "pt_1_t_b": {
        "time": [],
        "T1": [],
        "P1": [],
        "mass_flowrate_gmin": [],
        "heater_energy_kj": [],
    },

    "pt_2_t_b": {

        "time": [],
        "T1": [],
        "P1": [],
        "mass_flowrate_gmin": [],
        "heater_energy_kj": [],
    },

    "pt_1_t_c": {
        "time": [],
        "T1": [],
        "P1": [],
        "mass_flowrate_gmin": [],
        "heater_energy_kj": [],
    },

    "pt_2_t_c": {
        "time": [],
        "T1": [],
        "P1": [],
        "mass_flowrate_gmin": [],
        "heater_energy_kj": [],
    },

    "pt_1_t_d": {

        "time": [],
        "T1": [],
        "P1": [],
        "mass_flowrate_gmin": [],
        "heater_energy_kj": [],
    },

    "pt_2_t_d": {

        "time": [],
        "T1": [],
        "P1": [],
        "mass_flowrate_gmin": [],
        "heater_energy_kj": [],
    },

  
}

# Function to read data from a CSV file and populate the appropriate dictionary
def read_csv_data(filename, data_key):
    with open(filename, 'r') as file:
        cur_line_idx = 0
        for line in file:
            if cur_line_idx < 4:
                cur_line_idx += 1
                continue
            columns = line.split('	')

            data_dict[data_key]["time"].append(float(columns[0]))
            data_dict[data_key]["T1"].append(float(columns[1]))
            data_dict[data_key]["P1"].append(float(columns[5]))
            data_dict[data_key]["mass_flowrate_gmin"].append(float(columns[7]))
            data_dict[data_key]["heater_energy_kj"].append(float(columns[8]))

            cur_line_idx += 1


# Function to plot Pressure vs Time
def plotFillingSection(data_key, cutoff_index = -1):
    fig, ax = plt.subplots()
    ax.plot(data_dict[data_key]["time"][:cutoff_index], data_dict[data_key]["P1"][:cutoff_index], label='P1 (PSI)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure (PSI)')
    ax.legend()
    ax.set_title('Pressure vs Time')
    
    plt.show()

# Function to plot Mass Flowrate vs Time
def plotFlowrateFill(data_key, start_index, cutoff_index, first_x, second_x):
    fig, ax = plt.subplots()

    time = data_dict[data_key]["time"]
    mass_flowrate = data_dict[data_key]["mass_flowrate_gmin"]

    ax.plot(data_dict[data_key]["time"][start_index:cutoff_index], data_dict[data_key]["mass_flowrate_gmin"][start_index:cutoff_index], label='Mass Flowrate (g/min)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass Flowrate (g/min)')
    ax.legend()
    ax.set_title('Mass Flowrate vs Time, Part 1 Trial ' + data_key[-1])
    


    plt.axvline(x=first_x, color='r', linestyle='--')
    plt.axvline(x=second_x, color='r', linestyle='--')

    # Shade the area under the graph between first_x and second_x
    x = np.array(time[start_index:cutoff_index])
    y = np.array(mass_flowrate[start_index:cutoff_index])
    
    mask = (x >= first_x) & (x <= second_x)
    plt.fill_between(x[mask], y[mask], color='red', alpha=0.3)

    #numerically integrate the area under the graph, between first_x and second_x, keeping in mind that flow rate is in g/min and time is in seconds
    area = np.trapz(y[mask], x[mask]) / 60
    
    uncert_g = None

    #if trial is a or c, uncert is 0.06g
    #if trial is b or d, uncert ins 0.08g
    if data_key[-1] == 'a' or data_key[-1] == 'c':
        uncert_g = 0.06

    elif data_key[-1] == 'b' or data_key[-1] == 'd':
        uncert_g = 0.08


    #label the area
    plt.text(16, 8, "Area = " + str(round(area, 2)) + " +/- " + str(uncert_g) + " g", fontsize=12, color='red')

    #save in /Graphs/name
    plt.savefig('Graphs/' + data_key + "_mass_flowrate" + '.png')



    # Show the plot
    #plt.show()


    plt.clf()
    plt.cla()
    plt.close()


    
from scipy.ndimage import gaussian_filter1d
# Reading the data from the CSV files
read_csv_data('data/Lab 2 - Part1a', 'pt_1_t_a')
read_csv_data('data/Lab 2 - Part2a', 'pt_2_t_a')
read_csv_data('data/Lab 2 - Part1b', 'pt_1_t_b')
read_csv_data('data/Lab 2 - Part2b', 'pt_2_t_b')
read_csv_data('data/Lab 2 - Part1c', 'pt_1_t_c')
read_csv_data('data/Lab 2 - Part2c', 'pt_2_t_c')
read_csv_data('data/Lab 2 - Part1d', 'pt_1_t_d')
read_csv_data('data/Lab 2 - Part2d', 'pt_2_t_d')



#plot filling for first part
#plotFlowrateFill('pt_1_t_a', 100, 520, 10, 50)
#plotFlowrateFill('pt_1_t_b', 100, 800, 10, 80)
#plotFlowrateFill('pt_1_t_c', 120, 500, 12, 50)
#plotFlowrateFill('pt_1_t_d', 50, 650, 5, 65)

#exit(-1)

#next we need to plot the temperture vs time for each trial of part 2 and the kJ added, in two subplots on top of eachother
#plotting the temperature vs time for each trial of part 2

def plotTempAndHeaterEnergy(data_key, cutoff_index = -1):
    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(data_dict[data_key]["time"][:cutoff_index], data_dict[data_key]["T1"][:cutoff_index], label='T1 (C)')
    ax[0].set_ylabel('Temperature (C)')
    ax[0].legend()
    ax[0].set_title('Temperature vs Time, Part 2 Trial ' + data_key[-1])

    ax[1].plot(data_dict[data_key]["time"][:cutoff_index], data_dict[data_key]["heater_energy_kj"][:cutoff_index], label='Heater Energy (kJ)', color='r')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Heater Energy (kJ)')
    ax[1].legend()

    plt.savefig('Graphs/' + data_key + "_temp_and_heater_energy" + '.png')

    plt.show()


#plotTempAndHeaterEnergy('pt_2_t_a')
#plotTempAndHeaterEnergy('pt_2_t_b')
#plotTempAndHeaterEnergy('pt_2_t_c')
#plotTempAndHeaterEnergy('pt_2_t_d')

#exit(-1)

#c_v = heat_addition_rate / (mass_flowrate * delta_temp)

import numpy as np
import matplotlib.pyplot as plt



def plotTempAndHeaterEnergy(data_key, mass, sigma=100, cutoff_index=-1):
    # Apply Gaussian smoothing
    temp = gaussian_filter1d(np.array(data_dict[data_key]["T1"][:cutoff_index]), sigma=sigma)
    time = np.array(data_dict[data_key]["time"][:cutoff_index])
    #smooth time for consistency
    time = gaussian_filter1d(time, sigma=sigma)
    heater_energy = gaussian_filter1d(np.array(data_dict[data_key]["heater_energy_kj"][:cutoff_index]), sigma=sigma)

    # Calculate the derivative of temperature
    temp_derivative = np.diff(temp) / np.diff(time)

    # Calculate the derivative of heater energy (rate of heat addition)
    energy_derivative = np.diff(heater_energy) * 1000 / np.diff(time)  # Convert kJ to J and then to Watts


    realign = 80

    #add 50 nans to start of energy derivative to shift it
    energy_derivative = np.concatenate((np.full(realign, np.nan), energy_derivative))[:-realign]

    # Calculate specific heat capacity c_v at each point
    # Avoid division by zero in temperature difference
    #temp_diff = np.diff(temp)
    #non_zero_temp_diff = np.where(temp_diff != 0, temp_diff, np.nan)

    c_v = energy_derivative / (mass * temp_derivative)

    # Create subplots
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 10))

    # Temperature plot
    axs[0].plot(time, temp, label='T1 (C)')
    axs[0].set_ylabel('Temperature (C)')
    axs[0].legend()
    axs[0].set_title('Temperature and Heater Energy Analysis - ' + data_key)

    # Temperature derivative plot
    axs[1].plot(time[1:], temp_derivative, label='dT/dt (C/s)', color='orange')
    axs[1].set_ylabel('Rate of Temp Change (C/s)')
    axs[1].legend()

    # Heater energy plot
    axs[2].plot(time, heater_energy, label='Heater Energy (kJ)', color='r')
    axs[2].set_ylabel('Heater Energy (kJ)')
    axs[2].legend()

    # Heater energy derivative (rate of heat addition) plot
    axs[3].plot(time[1:], energy_derivative, label='dE/dt (W)', color='green')
    axs[3].set_ylabel('Rate of Heat Addition (W)')
    axs[3].legend()

    c_v_start_index = 150
    c_v_cutoff_index = 800

    # Specific heat capacity c_v plot
    axs[4].plot(time[c_v_start_index:c_v_cutoff_index], c_v[c_v_start_index:c_v_cutoff_index], label='c_v (J/g°C)', color='blue')
    axs[4].set_xlabel('Time (s)')
    axs[4].set_ylabel('c_v (J/g°C)')
    axs[4].legend()

    # Save and show the plot
    plt.tight_layout()
    plt.savefig('Graphs/' + data_key + "_temp_and_heater_energy_cv_analysis.png")
    plt.show()






#NOTE: gausien smoothing used since otherwise the derivatives are too noisy to make out important values
#plotTempAndHeaterEnergy('pt_2_t_a', 22.4)



def computeAirMass_g(t_initial, ambient_pressure = 101.325, gas_constant = 8.314, molar_mass = 28.97, v_cylinder = 0.0336):
    #t_initial in C
    #ambient pressure in kPa
    #gas constant in J/molK
    #molar mass in g/mol

    #convert ambient pressure to Pa
    ambient_pressure *= 1000

    #convert t_initial to K
    t_initial += 273.15

    #compute air mass
    air_mass = molar_mass * (ambient_pressure * v_cylinder) / (gas_constant * t_initial)
    return air_mass 




def plotTempAndHeaterEnergySimple(data_key, sigma=10, cutoff_index=-1, calc_start=None, calc_end=None, mass_g=None):
    # Apply Gaussian smoothing
    temp = gaussian_filter1d(np.array(data_dict[data_key]["T1"][:cutoff_index]), sigma=sigma)
    time = np.array(data_dict[data_key]["time"][:cutoff_index])
    heater_energy = gaussian_filter1d(np.array(data_dict[data_key]["heater_energy_kj"][:cutoff_index]), sigma=sigma)


    initial_air_mass = computeAirMass_g(temp[0])

    mass_added = mass_g


    mass_g = mass_g + initial_air_mass

    #print("Initial air mass: " + str(initial_air_mass) + "g")

    # Create subplots
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

    # Temperature plot
    axs[0].plot(time, temp, label='T1 (C)')
    axs[0].set_ylabel('Temperature (C)')
    axs[0].legend()
    axs[0].set_title('Temperature and Heater Energy Analysis - ' + "Trial " + data_key[-1])

    # Heater energy plot
    axs[1].plot(time, heater_energy, label='Heater Energy (kJ)', color='r')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Heater Energy (kJ)')
    axs[1].legend()

    if calc_start is not None and calc_end is not None and mass_g is not None:
        # Find indices for calc_start and calc_end
        start_index = np.searchsorted(time, calc_start)
        end_index = np.searchsorted(time, calc_end)

        # Initial and end temperatures and heater energies
        init_temp = temp[start_index]
        end_temp = temp[end_index]
        init_energy = heater_energy[start_index]
        end_energy = heater_energy[end_index]

        # Annotating the initial and end points
        axs[0].axhline(y=init_temp, color='green', linestyle='--')
        axs[0].axhline(y=end_temp, color='blue', linestyle='--')
        axs[1].axhline(y=init_energy, color='green', linestyle='--')
        axs[1].axhline(y=end_energy, color='blue', linestyle='--')

        axs[0].axvline(x=calc_start, color='gray', linestyle='--')
        axs[0].axvline(x=calc_end, color='gray', linestyle='--')
        axs[1].axvline(x=calc_start, color='gray', linestyle='--')
        axs[1].axvline(x=calc_end, color='gray', linestyle='--')

        # Calculate c_v and annotate
        delta_q = (end_energy - init_energy) * 1000  # Convert kJ to J
        delta_t = end_temp - init_temp
        c_v = delta_q / (mass_g * delta_t) if delta_t != 0 else float('inf')


        delta_energy_uncertainty = 0.005  # Uncertainty in energy in kJ
        delta_temp_uncertainty = 0.05  # Uncertainty in temperature in C

        #mass uneccertainty is 0.06g for a and c, 0.08g for b and d
        mass_uncertainty = 0.06 if data_key[-1] == 'a' or data_key[-1] == 'c' else 0.08



            # Calculate the uncertainties
        delta_energy = abs(end_energy - init_energy) * 1000  # Convert kJ to J
        delta_temp = abs(end_temp - init_temp)
        relative_energy_uncertainty = delta_energy_uncertainty / delta_energy if delta_energy != 0 else 0
        relative_temp_uncertainty = delta_temp_uncertainty / delta_temp if delta_temp != 0 else 0
        relative_mass_uncertainty = mass_uncertainty / mass_g

        # Calculate the uncertainty in c_v
        delta_c_v = c_v * np.sqrt(relative_energy_uncertainty**2 + relative_temp_uncertainty**2 + relative_mass_uncertainty**2)

        # Annotate c_v with uncertainty
        axs[0].annotate(f'c_v: {c_v:.2f} ± {delta_c_v:.2f} J/g°C', xy=(calc_end, end_temp), xytext=(calc_end + 5, end_temp - 3.5),
                        arrowprops=dict(facecolor='black', shrink=0.05), fontsize=16)




        #annotate to the right of the end Temp point,  t=70, T=25, big font size
        #axs[0].annotate(f'c_v: {c_v:.2f} J/g°C', xy=(calc_end, end_temp), xytext=(calc_end + 5, end_temp - 3.5),
        #                arrowprops=dict(facecolor='black', shrink=0.05), fontsize=16)

        #below that annotated (mass added, iniital mass of air) in g as label
        axs[0].annotate(f'(Using m_initial + m_added = {initial_air_mass:.2f} +  {mass_added:.2f} = {initial_air_mass+mass_added:.2f}g)', xy=(calc_end + 5, end_temp - 5.3))




    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.2, wspace=0.2)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig('Graphs/' + "_temp_and_heater_energy_analysis_trial_" + data_key + ".png", bbox_inches='tight')

    #plt.show()

#plotTempAndHeaterEnergySimple('pt_2_t_a', calc_start = 20, calc_end = 50, mass_g = 23.57, cutoff_index = 3000)
#plotTempAndHeaterEnergySimple('pt_2_t_b', calc_start = 15, calc_end = 40, mass_g = 43.3, cutoff_index = 3000)
#plotTempAndHeaterEnergySimple('pt_2_t_c', calc_start = 15, calc_end = 40, mass_g = 23.17, cutoff_index = 3000)
#plotTempAndHeaterEnergySimple('pt_2_t_d', calc_start = 15, calc_end = 40, mass_g = 40.95, cutoff_index = 3000)

#exit(-1)

def plotHeaterTempMaintain(data_key, start_idx, end_idx, sigma=10):
    # Extract relevant data within the specified range and apply Gaussian smoothing
    temp = gaussian_filter1d(np.array(data_dict[data_key]["T1"]), sigma=sigma)[start_idx:end_idx]
    time = np.array(data_dict[data_key]["time"])[start_idx:end_idx]
    heater_energy = gaussian_filter1d(np.array(data_dict[data_key]["heater_energy_kj"]), sigma=sigma)[start_idx:end_idx]

    # Calculate the rate of heat addition
    # Assuming time is in seconds and energy in kJ, we convert energy to J for rate calculation
    time_diff = np.diff(time)
    energy_diff = np.diff(heater_energy * 1000)  # Convert kJ to J
    heat_rate = energy_diff / time_diff  # Rate in Watts (J/s)

    # Time array for heat rate should be the midpoint of the time intervals
    time_heat_rate = time[:-1] + time_diff / 2

    # Create subplots
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    # Temperature plot
    axs[0].plot(time, temp, label='Temperature (°C)')
    axs[0].set_ylabel('Temperature (°C)')
    axs[0].legend()

    # Heater energy plot
    axs[1].plot(time, heater_energy, label='Heater Energy (kJ)', color='r')
    axs[1].set_ylabel('Heater Energy (kJ)')
    axs[1].legend()

    # Heat rate plot
    axs[2].plot(time_heat_rate, heat_rate, label='Heat Rate (W)', color='green')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Heat Rate (W)')
    axs[2].legend()

    # Setting titles
    axs[0].set_title('Temperature Maintenance Analysis - ' + "Trial " + data_key[-1])

    #plot horziontal line at average heat rate
    avg_heat_rate = np.average(heat_rate)
    axs[2].axhline(y=avg_heat_rate, color='gray', linestyle='--')
    #annotate
    axs[2].annotate(f'Average Heat Rate: {avg_heat_rate:.2f} W', xy=(time_heat_rate[0] + 30, avg_heat_rate - 25), fontsize=16)

    # Save and show the plot
    plt.tight_layout()



    plt.rcParams["figure.figsize"] = (6, 6)


    plt.savefig('Graphs/' + "heater_temp_maintenance_" + data_key + ".png")

    #set plot size
    #set aspect ratio to square



    #plt.show()

#plotHeaterTempMaintain('pt_2_t_a', 2000, 4000)
#plotHeaterTempMaintain('pt_2_t_b', 2000, 4000)
#plotHeaterTempMaintain('pt_2_t_c', 2000, 4000)
#plotHeaterTempMaintain('pt_2_t_d', 2000, 4000)

import math

import math

def calculate_heat_transfer_with_uncertainty(inner_wall_temp, net_heat_transfer, net_heat_transfer_uncertainty):
    # Constants with their uncertainties
    k_acrylic = 0.185  # Thermal conductivity in W/(m*K)
    k_acrylic_uncertainty = 0.015  # Uncertainty in thermal conductivity

    l = 0.2858  # Length in meters
    l_uncertainty = 0.000001  # Uncertainty in length

    r1 = 0.09208  # Inner radius in meters
    r1_uncertainty = 0.0006  # Uncertainty in inner radius

    r2 = 0.1016  # Outer radius in meters
    r2_uncertainty = 0.001  # Uncertainty in outer radius

    outer_wall_temp = 18.5  # Outer wall temperature in degrees Celsius
    outer_wall_temp_uncertainty = 0.05  # Uncertainty in outer wall temperature

    inner_wall_temp_uncertainty = 0.05  # Uncertainty in inner wall temperature

    # Calculate the temperature difference and its uncertainty
    delta_T = inner_wall_temp - outer_wall_temp
    delta_T_uncertainty = math.sqrt(inner_wall_temp_uncertainty**2 + outer_wall_temp_uncertainty**2)

    # Calculate heat transfer through the walls using the given formula
    heat_transfer_walls = 2 * k_acrylic * math.pi * l * (delta_T / math.log(r2 / r1))

    # Uncertainty propagation
    partial_k = 2 * math.pi * l * (delta_T / math.log(r2 / r1))
    partial_l = 2 * k_acrylic * math.pi * (delta_T / math.log(r2 / r1))
    partial_delta_T = 2 * k_acrylic * math.pi * l / math.log(r2 / r1)
    partial_r1 = -2 * k_acrylic * math.pi * l * delta_T / (r1 * (math.log(r2 / r1)**2))
    partial_r2 = 2 * k_acrylic * math.pi * l * delta_T / (r2 * (math.log(r2 / r1)**2))

    #print("partial_k: " + str(partial_k))
    #print("partial_l: " + str(partial_l))
    #print("partial_delta_T: " + str(partial_delta_T))
    #print("partial_r1: " + str(partial_r1))
    #print("partial_r2: " + str(partial_r2))

    heat_transfer_walls_uncertainty = math.sqrt((partial_k * k_acrylic_uncertainty)**2 +
                                                (partial_l * l_uncertainty)**2 +
                                                (partial_delta_T * delta_T_uncertainty)**2 +
                                                (partial_r1 * r1_uncertainty)**2 +
                                                (partial_r2 * r2_uncertainty)**2)

    # Calculate heat transfer through the plates and its uncertainty
    heat_transfer_plates = net_heat_transfer - heat_transfer_walls
    heat_transfer_plates_uncertainty = math.sqrt(net_heat_transfer_uncertainty**2 + heat_transfer_walls_uncertainty**2)

    return heat_transfer_walls, heat_transfer_walls_uncertainty, heat_transfer_plates, heat_transfer_plates_uncertainty



#w, w_u, p, p_u = calculate_heat_transfer_with_uncertainty(40, 133.34, 0.005)
#w, w_u, p, p_u = calculate_heat_transfer_with_uncertainty(40, 143.85, 0.005)
#w, w_u, p, p_u = calculate_heat_transfer_with_uncertainty(60, 219.51, 0.005)
#w, w_u, p, p_u = calculate_heat_transfer_with_uncertainty(60, 264.83, 0.005)

#print("Heat transfer through walls: " + str(w) + " +/- " + str(w_u))
#print("Heat transfer through plates: " + str(p) + " +/- " + str(p_u))


import math

def calculate_P2_and_uncertainty(p2, T2, sigma_p2, sigma_T2):
    # Constants
    P1 = 0.07457  # in Watts
    p1 = 101.325  # in kPa
    T1_Celsius = 25  # in degrees Celsius
    T1_Kelvin = T1_Celsius + 273.15  # Convert to Kelvin

    # Convert T2 from Celsius to Kelvin
    T2_Kelvin = T2 + 273.15

    # Calculate P2
    P2 = 9.261 * P1 * (p2 * T1_Kelvin) / (p1 * T2_Kelvin)

    # Calculate the partial derivatives
    partial_p2 = 9.261 * P1 * T1_Kelvin / (p1 * T2_Kelvin)
    partial_T2 = -9.261 * P1 * p2 * T1_Kelvin / (p1 * T2_Kelvin**2)

    # Calculate uncertainty in P2
    sigma_P2 = math.sqrt((partial_p2 * sigma_p2)**2 + (partial_T2 * sigma_T2)**2)

    return P2, sigma_P2

#P2_result, sigma_P2_result = calculate_P2_and_uncertainty(270.3 , 19.6 , 0.1, 0.1)
#P2_result, sigma_P2_result = calculate_P2_and_uncertainty(473.6 , 23.9 , 0.1, 0.1)
#P2_result, sigma_P2_result = calculate_P2_and_uncertainty(260.6 , 31.1 , 0.1, 0.1)
P2_result, sigma_P2_result = calculate_P2_and_uncertainty(475.1 , 31.9 , 0.1, 0.1)



print(f"P2: {P2_result} Watts, Uncertainty in P2: {sigma_P2_result} Watts")
