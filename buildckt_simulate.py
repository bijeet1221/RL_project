import subprocess
import os
'''
input : list of list
         each list is [component_type, value, node1, node2]
        component_type : "res", "cap", "ind"
        value, node1, node2 are integers
        node value  is
                0 for ground
                1 for Vin
-You don't need to have node1 and node2 sorted order, they can be in 
    any order like ["res", 1000, 2, 1] or ["res", 1000, 1, 2], code takes care of it

-Code takes care even if u skip a node like  
 nodes 1 2 3 4  
 nodes 1 2 4 8
 if component connections are relatively same, the same circuit is generated
'''


def build_circuit(component_list, filename="generated_circuit.asc"):
    print("Building circuit...")
    print(component_list)
    nodes = set()
    for comp in component_list:
        n1, n2 = str(comp[2]), str(comp[3])
        if n1 != '0': nodes.add(n1)
        if n2 != '0': nodes.add(n2)

    sorted_nodes = sorted(nodes, key=int)
    node_index_map = {node: idx for idx, node in enumerate(sorted_nodes)}

    wires = []
    for idx in range(len(sorted_nodes)):
        x = idx * 112
        wires.append(f"WIRE {x} -168 {x} {(len(component_list))* 80}")

    component_blocks = []
    wire_blocks = []
    flag_blocks = []
    delta = 0 
    counts = {"res": 0, "cap": 0, "ind": 0, "voltage": 0}
    for comp in component_list:
        ctype, value, node1, node2 = comp
        node1, node2 = str(node1), str(node2)

        if ctype not in counts:
            counts[ctype] = 0
        counts[ctype] += 1
        inst_name = f"{ctype[0].upper()}{counts[ctype]}"

        if node1 != '0' and node2 != '0':
            idx1 = node_index_map[node1]
            idx2 = node_index_map[node2]
            smaller_idx = min(idx1, idx2)
        else:
            non_zero_node = node2 if node1 == '0' else node1
            smaller_idx = node_index_map[non_zero_node]

        x = smaller_idx * 112 + (64 if ctype == "cap" else 96)
        y = -16 + 80 * delta
        delta += 1

        component_blocks.extend([
            f"SYMBOL {ctype} {x} {y} R90",
            f"SYMATTR InstName {inst_name}",
            f"SYMATTR Value {value}"
        ]) # adding a component

        if node1 == '0' or node2 == '0':
            offset = 0 if ctype == "cap" else 16
            flag_blocks.append(f"FLAG {x - offset} {y + 16} 0") 
            #directly connecting to ground instead of wires for node=0
        else:
            x1 = x - (0 if ctype == "cap" else 16)
            x2 = max(idx1, idx2) * 112
            wire_blocks.append(f"WIRE {x1} {y+16} {x2} {y+16}")

    with open(filename, 'w') as f:
        # these are defult commands for initialization and placing voltage source, 
        # source name is AC1 and amplitude is 1V
        f.write("Version 4\nSHEET 1 880 680\n")
        f.write("SYMBOL voltage -272 -176 R0\n")
        f.write("WINDOW 123 0 0 Left 0\n")
        f.write("WINDOW 39 0 0 Left 0\n")
        f.write("SYMATTR InstName AC1\n")
        f.write("SYMATTR Value AC 1\n")
        f.write("FLAG -272 -80 0\n")
        f.write("WIRE 0 -160 -272 -160\n")
        f.write("TEXT -380 328 Left 2 !.ac oct 10 1 1000\n")
        f.write(f"FLAG {112} {(-144)} Vout\n")
        # this like is for simulation setting, its ac analysis, 10 points per decade
        # from 1Hz to 1000Hz
        for wire in wires:
            f.write(wire + "\n")
        for line in component_blocks:
            f.write(line + "\n")
        for line in wire_blocks:
            f.write(line + "\n")
        for line in flag_blocks:
            f.write(line + "\n")

    print(f".asc file generated: {filename}")


components = [
    ["res", 1000, 2, 1],
    ["cap", 10e-5, 2, 0],
    ["ind", 0.1, 2, 0],
    ["res", 1000, 2, 1],
    ["cap", 10e-5, 2, 0],
    ["ind", 0.2, 2, 0],
    ["cap", 10e-5, 2, 3],
    ["ind", 0.2, 3, 0],
]

def run_simulation(circuit_path, ltspice_path):
    # Add -b (batch) and -run to ensure simulation runs in the background without waiting for user input
    command = [ltspice_path, "-b", "-run", circuit_path]
    # print(f"Running LTSpice simulation with command: {' '.join(command)}")

    try:
        # Run the LTSpice simulation and capture stdout and stderr
        result = subprocess.run(command, check=True, capture_output=True, text=True)

        # Print the output from LTSpice (stdout and stderr)
        if result.stdout:
            print("LTSpice stdout:", result.stdout)
        if result.stderr:
            print("LTSpice stderr:", result.stderr)
            
        # Check the return code to see if the simulation was successful
        if result.returncode == 0:
            print("Simulation completed successfully.")
        else:
            print(f"Simulation failed with return code: {result.returncode}")
            print("Error:", result.stderr)

        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running LTSpice: {e}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        return None

# def plot_response(raw_file_path):
#     from PyLTSpice import RawRead
#     import matplotlib.pyplot as plt
#     import numpy as np

#     # Load the raw file
#     ltr = RawRead("custom_rl_circuit.raw")

#     # Print all available traces (just to verify)
#     print("[*] Available traces:", ltr.get_trace_names())

#     # Get frequency and voltages
#     freq = ltr.get_trace("frequency")
#     vout = ltr.get_trace("V(Vout)")  # Output node (after filter)

#     # Extract data
#     f_vals = freq.get_wave(0)
#     vout_vals = vout.get_wave(0)

#     # Compute gain magnitude (|Vout/Vin|)
#     gain = np.abs(vout_vals )

#     # Compute phase (angle of Vout/Vin)
#     # phase = np.angle(vout_vals, deg=True)  # Phase in degrees

#     # Plot Gain (Magnitude)
#     plt.figure(figsize=(10, 6))
#     plt.subplot(1, 1, 1)
#     plt.semilogx(f_vals, gain, color='blue')
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Gain")
#     plt.title("Frequency Response (Gain)")


#     # Tight layout for better visualization
#     plt.tight_layout()
#     plt.show()

gain = 0


resistor_values = [1000, 2000, 3000, 4000, 5000]  # Example resistor values in Ohms
capacitor_values = [3*1e-6, 2*1e-6, 1e-6, 4*1e-6]  # Example capacitor values in Farads
inductor_values = [0.1, 0.5, 1.0, 2.0]  # Example inductor values in Henrys

def generate_random_circuit():
    import random
    circuit = []
    
    # Generate a random number of components for this circuit
    num_components = random.randint(3, 10)  # You can adjust this range as needed
    
    # List of possible component types
    component_types = ["res", "cap", "ind"]
    
    for _ in range(num_components):
        # Randomly choose a component type
        component_type = random.choice(component_types)
        
        # Generate a random value for the component from the predefined sets
        if component_type == "res":
            value = random.choice(resistor_values)  # Randomly choose a resistor value
        elif component_type == "cap":
            value = random.choice(capacitor_values)  # Randomly choose a capacitor value
        elif component_type == "ind":
            value = random.choice(inductor_values)  # Randomly choose an inductor value
        
        # Randomly choose two nodes (assuming a simple circuit with nodes numbered 1 to 10)
        node1 = random.randint(0, num_components)  # Randomly choose a node (0 for ground, 1-5 for other nodes)
        node2 = random.randint(0, num_components)  # Randomly choose another node (0 for ground, 1-5 for other nodes)

        flag =0 
        if node1 == 2:
            flag = 1
        if node2 == 2:
            flag = 1

        # Make sure node1 and node2 are not the same
        while node1 == node2:
            node2 = random.randint(0,  (num_components))
        
        # Append the component to the circuit
        if node1!=0 or node2!=0:
            circuit.append([component_type, value, node1, node2])
    
    return circuit

def plot_response(raw_file_path, output_csv="frequency_response.csv"):
    from PyLTSpice import RawRead
    import numpy as np
    import csv

    # Load the raw file
    ltr = RawRead(raw_file_path)

    # Get frequency and voltages
    freq = ltr.get_trace("frequency")
    vout = ltr.get_trace("V(Vout)")  # Output node (after filter)

    # Extract data
    f_vals = freq.get_wave(0)
    vout_vals = vout.get_wave(0)

    # Compute gain magnitude (|Vout|)
    gain = np.abs(vout_vals)

    # Save to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frequency", "Gain"])
        for f, g in zip(f_vals, gain):
            writer.writerow([f, g])

    print(f"Frequency response saved to '{output_csv}'")
    return gain

import numpy as np

def get_reward(gain, target):
    gain1 = np.array(gain).reshape(1, -1)  # Reshape to 2D
    target1 = np.array(target).reshape(1, -1)  # Reshape to 2D
    target1 = np.array(target1, dtype=np.float64)
    
    corr = np.corrcoef(gain1, target1)[0, 1]
    reward =-1
    if corr >= 0.8:
        reward = 10
    return reward



def read_csv(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    return df.values.flatten()

csv_read = read_csv("target.csv")
target = csv_read[1::2]  



build_circuit(components, "custom_rl_circuit.asc")
run_simulation("custom_rl_circuit.asc", r"C:\Users\lokes\Desktop\LTspice\LTspice.exe")
gain = plot_response("custom_rl_circuit.raw", "target.csv")
print(len(gain), len(target))
# print(gain)
# print("egeg")
# print(target)
print(get_reward(gain, target))


circuits = []
for _ in range(1200):
    circuit = generate_random_circuit()
    circuits.append(circuit)

for circuit in circuits:
    print(circuit)
    print(len(circuit))
    print("\n")

for circuit in circuits:
    try:
        circuitname = f"custom_rl_circuit_{circuits.index(circuit)}.asc"
        build_circuit(circuit, circuitname)
        
        # Attempt to run the LTSpice simulation
        try:
            run_simulation(circuitname, r"C:\Users\lokes\Desktop\LTspice\LTspice.exe")
        except Exception as e:
            print(f"Error running LTSpice for circuit {circuits.index(circuit)}: {e}")
            continue  # Skip this circuit if LTSpice fails
        
        # Set the CSV file name for the target
        csvname = f"target_{circuits.index(circuit)}.csv"
        print(f"Processing {csvname}")
        
        # Attempt to plot the response and process it
        try:
            gain = plot_response(f"custom_rl_circuit_{circuits.index(circuit)}.raw", csvname)
            print(get_reward(gain, target))
        except Exception as e:
            print(f"Error processing response for circuit {circuits.index(circuit)}: {e}")
            continue  # Skip this circuit if response plotting fails
        
    except Exception as e:  # Catching any other exceptions
        print(f"Unexpected error processing circuit {circuits.index(circuit)}: {e}")
        continue  # Skip the circuit and continue to the next one

    
