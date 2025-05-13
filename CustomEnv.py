# # -*- coding: utf-8 -*-
# """
# Created on Sun Apr 13 17:18:42 2025

# @author: Bijeet Basak, Lokesh Kumar, Yash Sengupta
# """

# import gymnasium as gym  # The base class for any RL environment
# from gymnasium import spaces  # Helps us in defining action and observation spaces
# import numpy as np
# import networkx as nx  # A graph library to model our circuits as graphs
# import read_csv_and_feature_points as rd
# import subprocess
# import os


# simulation_counter = 0
# unique_counter = 0
# good_circuits = []

# uniquecorr = set()

# class CircuitEnv(gym.Env):  # Custom Environment


#     def __init__(self, max_components=15, value_buckets=5):
        
#         # print("Init")
#         super(CircuitEnv, self).__init__()

#         self.max_components = max_components  # Maximum number of components possible in a circuit.
#         self.value_buckets = value_buckets  # Maximum number of possible values each compomnent can have. 
#         self.node_counter = 2  # 0 = GND, 1 = VDD --> Each circuit starts with 2 nodes (GND and VDD)
#         self.episode_number = 0
#         self.freq_points = 10
#         self.corr = 0
        
#         # You can define custom buckets later for R, L, C
#         self.bucket_ranges = {
#             0: np.array([1000, 2000, 3000, 4000, 5000]),     # R: 1Ω to 1MΩ
#             2: np.array([0.1, 0.5, 1.0, 2.0, 2.5]),   # L: 1uH to 1mH
#             1: np.array([3*1e-6, 2*1e-6, 1e-6, 4*1e-6, 5*1e-6])   # C: 1pF to 1uF
#         }
        
#         self.component_list = ["res", "cap", "ind"]

#         # Action = [component_type, value_index, node1, node2]
#         # 0 --> R, 1 --> L, 2 --> C  component_type
#         # Each value_bucket contains a list of values for each component. The action space will specify the index of the value_buckets[]. Based on the index we will take the value from the bucket.
#         # The next two vlaues are the node indices (possible values where we can put a component)
#         self.action_space = spaces.MultiDiscrete([3, value_buckets, max_components + 2, max_components + 2])

#         # Observation: padded list of components: [type_id, value_idx, node1, node2]
#         self.observation_space = spaces.Dict({
#             "components": spaces.Box(-1, 1, shape=(self.max_components, 4), dtype=np.float32),
#             "target_response": spaces.Box(0, 1, shape=(2, 10), dtype=np.float32),
#             })
        
#         # Creating the train data object
#         folder_path = "train_graphs/csvs"
#         self.processor = rd.CSVProcessor(folder_path)
        
#         self.LTSpice = r"C:\Users\lokes\Desktop\LTspice\LTspice.exe"

#         self.reset()
#         # print("Init Done")
    
#     def update_Graph(self, circuit_graph, new_node_value, old_node_value):
        
#         # print(f"Components before updaing Graph:{self.components}")
#         # self.components = []
#         # print(f"New_node={new_node_value}, Old_node={old_node_value}")
#         new_components = []
#         node_value = None
#         for item in self.components:
#             if item[3] == old_node_value:
#                 node_value = new_node_value
#             else:
#                 node_value = item[3]
#             new_components.append((item[0], item[1], item[2], node_value))
#         self.components = new_components
#         new_graph = nx.DiGraph()
#         new_graph.add_nodes_from([0, 1])
#         for u, v, data in self.G.edges(data=True):
#             edge_type = data['type']
#             edge_value = data['value']
#             # print(f"{edge_value}")
#             # print(f"u={u}, v={v}")
#             if v == old_node_value:
#                 # print("v update")
#                 v = new_node_value
#                 # print(u, v)
                
#             new_graph.add_edge(u, v,  type=edge_type, value=edge_value)
#             # self.components.append((edge_type, edge_value, u, v))
#             # print(f"Components after updaing Graph:{self.components}")
                
#             # Handle node assignment
#             for node in [u, v]:
#                 if node not in new_graph.nodes:
#                     new_graph.add_node(node)  # Add the node to the graph representing the circuit
#         return new_graph
                
#     # LT Spice Interfacing Code
#     def build_circuit(self, component_list, filename="generated_circuit.asc"):
#         print("Building Circuit")
#         print(f"Component list length: {len(component_list)}")
#         nodes = set()
#         for comp in component_list:
#             n1, n2 = str(comp[2]), str(comp[3])
#             if n1 != '0': nodes.add(n1)
#             if n2 != '0': nodes.add(n2)

#         sorted_nodes = sorted(nodes, key=int)
#         node_index_map = {node: idx for idx, node in enumerate(sorted_nodes)}

#         wires = []
#         for idx in range(len(sorted_nodes)):
#             x = idx * 112
#             wires.append(f"WIRE {x} -168 {x} {(len(component_list))* 80}")

#         component_blocks = []
#         wire_blocks = []
#         flag_blocks = []
#         delta = 0 
#         counts = {"res": 0, "cap": 0, "ind": 0, "voltage": 0}
#         for comp in component_list:
#             ctype, value, node1, node2 = comp
#             node1, node2 = str(node1), str(node2)

#             if ctype not in counts:
#                 counts[ctype] = 0
#             counts[ctype] += 1
#             inst_name = f"{ctype[0].upper()}{counts[ctype]}"

#             if node1 != '0' and node2 != '0':
#                 idx1 = node_index_map[node1]
#                 idx2 = node_index_map[node2]
#                 smaller_idx = min(idx1, idx2)
#             else:
#                 non_zero_node = node2 if node1 == '0' else node1
#                 smaller_idx = node_index_map[non_zero_node]

#             x = smaller_idx * 112 + (64 if ctype == "cap" else 96)
#             y = -16 + 80 * delta
#             delta += 1

#             component_blocks.extend([
#                 f"SYMBOL {ctype} {x} {y} R90",
#                 f"SYMATTR InstName {inst_name}",
#                 f"SYMATTR Value {value}"
#             ]) # adding a component

#             if node1 == '0' or node2 == '0':
#                 offset = 0 if ctype == "cap" else 16
#                 flag_blocks.append(f"FLAG {x - offset} {y + 16} 0") 
#                 #directly connecting to ground instead of wires for node=0
#             else:
#                 x1 = x - (0 if ctype == "cap" else 16)
#                 x2 = max(idx1, idx2) * 112
#                 wire_blocks.append(f"WIRE {x1} {y+16} {x2} {y+16}")

#         with open(filename, 'w') as f:
#             # these are defult commands for initialization and placing voltage source, 
#             # source name is AC1 and amplitude is 1V
#             f.write("Version 4\nSHEET 1 880 680\n")
#             f.write("SYMBOL voltage -272 -176 R0\n")
#             f.write("WINDOW 123 0 0 Left 0\n")
#             f.write("WINDOW 39 0 0 Left 0\n")
#             f.write("SYMATTR InstName AC1\n")
#             f.write("SYMATTR Value AC 1\n")
#             f.write("FLAG -272 -80 0\n")
#             f.write("WIRE 0 -160 -272 -160\n")
#             f.write("TEXT -380 328 Left 2 !.ac oct 10 1 1000\n")
#             f.write(f"FLAG {112} {(-144)} Vout\n")
#             # this like is for simulation setting, its ac analysis, 10 points per decade
#             # from 1Hz to 1000Hz
#             for wire in wires:
#                 f.write(wire + "\n")
#             for line in component_blocks:
#                 f.write(line + "\n")
#             for line in wire_blocks:
#                 f.write(line + "\n")
#             for line in flag_blocks:
#                 f.write(line + "\n")

#         print(f".asc file generated: {filename}")
    
#     def run_simulation(self, circuit_path, ltspice_path):
#         global simulation_counter
#         simulation_counter += 1
#         print(f"simulation_counter: {simulation_counter}")
#         counter_file = "simulation_counter.txt"
#         with open(counter_file, "w") as f:
#             f.write(str(simulation_counter))
#         # Add -b (batch) and -run to ensure simulation runs in the background without waiting for user input
#         command = [ltspice_path, "-b", "-run", circuit_path]
#         # print(f"Running LTSpice simulation with command: {' '.join(command)}")

#         try:
#             # Run the LTSpice simulation and capture stdout and stderr
#             result = subprocess.run(command, check=True, capture_output=True, text=True)

#             # Print the output from LTSpice (stdout and stderr)
#             if result.stdout:
#                 print("LTSpice stdout:", result.stdout)
#             if result.stderr:
#                 print("LTSpice stderr:", result.stderr)
                
#             # Check the return code to see if the simulation was successful
#             if result.returncode == 0:
#                 print("Simulation completed successfully.")
#             else:
#                 print(f"Simulation failed with return code: {result.returncode}")
#                 print("Error:", result.stderr)

#             return result
#         except subprocess.CalledProcessError as e:
#             print(f"Error running LTSpice: {e}")
#             print(f"Output: {e.output}")
#             print(f"Error: {e.stderr}")
#             return None
        
    
    
    
#     def plot_response(self, raw_file_path, output_csv="frequency_response.csv"):
#         from PyLTSpice import RawRead
#         import numpy as np
#         import csv

#         # Load the raw file
#         ltr = RawRead(raw_file_path)

#         # Get frequency and voltages
#         freq = ltr.get_trace("frequency")
       
#         vout = ltr.get_trace("V(Vout)")  # Output node (after filter)

#         # Extract data
#         f_vals = freq.get_wave(0)
#         vout_vals = vout.get_wave(0)

#         # Compute gain magnitude (|Vout|)
#         gain = np.abs(vout_vals)

#         # Save to CSV
#         with open(output_csv, mode='w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(["Frequency", "Gain"])
#             for f, g in zip(f_vals, gain):
#                 writer.writerow([f, g])

#         print(f"Frequency response saved to '{output_csv}'")
#         return gain
    
#     # LT Spice Interfacing Code ends...


#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
        
#         self.components = []  # List of components as tuples (type, value_idx, n1, n2)
#         self.G = nx.DiGraph()
#         self.G.add_nodes_from([0, 1])  # 0 = GND, 1 = VDD
#         self.node_counter = 2
        
        
#         self.norm_avgs, self.norm_slopes, self.target = self.processor.process_next()
#         self.current_file = self.processor.files[self.processor.index - 1]
        
#         self.episode_number += 1
        
#         obs = self.get_observation()
        
#         return obs, {}

#     def step(self, action):  # This method is called everytime the agent takes an action
        
#         # print("In Step")
    
#         new_node_flag = False # This tells if a new node is added in the circuit
#         old_node = None # This keeps track of the old node value which now becomes the new node. Useful for updating the graph of nodes.
#         # Action unpacking
#         comp_type, val_idx, node1, node2 = action
        
#         # Is the action valid in the current state or not?
#         # This ensures that that there are no floating nodes in the circuit
        
#         valid_nodes = len(self.G.nodes) # This gives me the current number of nodes in the circuit as well as the value of the next node.
#         # print(f"Valid Node ={valid_nodes}")
#         # You can define: current_node_limit = len(valid_nodes)
#         if node1 > valid_nodes or node2 > (valid_nodes - 1):
#             # Invalid node selection
#             return self.get_observation(), -5, False, False, {}
        
#         # If both nodes are same, that means one extra node is required to be added
#         if node1 == node2:
            
#             old_node = node1
#             node1 = valid_nodes
#             new_node_flag = True
            
#         # value = self.get_value(comp_type, val_idx)  # Based on the val_idx and the comp_type, extract the value of the component

#         # Handle new node assignment
#         for node in [node1, node2]:
#             if node not in self.G.nodes:
#                 self.G.add_node(node)  # Add the node to the graph representing the circuit

#         # print("Adding")
#         # Add component to circuit
        
        
#         # Check if new_node is there. If it is we need to update the entire graph
#         if new_node_flag == True:
#             # print("Updating Graph")
#             self.G = self.update_Graph(self.G, node1, old_node)
            
#         self.components.append((comp_type, val_idx, node1, node2))
        
#         # print("Adding Edge")
        
#         self.G.add_edge(node1, node2, type=comp_type, value=val_idx)
        
        
        
#         # Running the simulation and getting the result from LT Spice
#         components = []
#         gain = []
#         # print(self.bucket_ranges)
        
#         for item in self.components:
#             # print(self.component_list.index(item[0]))
#             # print(self.get_value((item[0]), item[1]))
#             components.append([self.component_list[item[0]], self.get_value((item[0]), item[1]), item[2], item[3]])
            
#         isthere2 = False
#         node = list(self.G.nodes)
#         for i in node:
#             if i == 2:
#                 isthere2 = True
#                 break
#         for i in range(len(components)):
#             if components[i][2] == 2 or components[i][3] == 2:
#                 isthere2 = True
#                 break

#         nodeset = set()
#         for i in components:
#             nodeset.add(i[2])
#             nodeset.add(i[3])

#         print(isthere2)
#         print("isthere2")
#         if len(components)>2 and len(nodeset) > 2 and isthere2:
#             # print(components)
#             self.build_circuit(components, "custom_rl_circuit.asc")
#             self.run_simulation("custom_rl_circuit.asc", self.LTSpice)
#             gain = self.plot_response("custom_rl_circuit.raw", "target.csv")
        

#         obs = self.get_observation()
#         reward = self.get_reward(gain, self.target)
#         done = self.is_done()

#         return obs, reward, done, False, {}

#     def get_observation(self):
#         obs = np.full((self.max_components, 4), -1, dtype=np.float32)

#         for i, (ctype, v_idx, n1, n2) in enumerate(self.components[-self.max_components:]):
#             value = self.get_value(ctype, v_idx)  # Based on the val_idx and the comp_type, extract the value of the component
#             obs[i] = [
#                 ctype / 2,              # Normalize type (0,1,2)
#                 value / (self.value_buckets - 1),  # Normalize value index
#                 n1 / (self.max_components + 1),
#                 n2 / (self.max_components + 1)
#             ]
#         target = np.stack(
#             [self.norm_avgs, self.norm_slopes], axis=0
#         ).astype(np.float32)
#         return {
#             "components": obs,
#             "target_response": target
#         }

#     def get_value(self, ctype, v_idx):
        
#         return self.bucket_ranges[ctype][v_idx]

#     def get_reward(self, gain, target):
        
#         if len(gain) == 0:
#             return -1
#         gain1 = np.array(gain).reshape(1, -1)  # Reshape to 2D
#         target1 = np.array(target).reshape(1, -1)  # Reshape to 2D
#         target1 = np.array(target1, dtype=np.float64)
        
#         self.corr = np.corrcoef(gain1, target1)[0, 1]
#         reward =-len(self.components)
#         if self.corr >= 0.8:
#             reward = 10
#         return reward

#     def is_done(self):
#         global unique_counter
#         global good_circuits
#         if(self.corr >= 0.8):
#             print("correlation: " + str(self.corr))
#             uniquecorr.add(self.corr)
#             print(unique_counter)
#             good_circuits.append(self.components)
#             # remove duplicated in good_circuits
#             good_circuits = [list(x) for x in set(tuple(x) for x in good_circuits)]
#             print(f"Good circuits: {len(good_circuits)}")
#             print("Unique correlation: " + str(len(uniquecorr)))
#             wirter = open("good_circuit.txt", "a")
#             for i in good_circuits:
#                 wirter.write(str(i) + "\n")
#             wirter.close()
#             # self.build_circuit(self.components, "good_circuit.asc")
#             unique_counter += 1
#         if(len(self.components) >= self.max_components):
#             print("Max components reached")
#             # print(unique_counter)
#             # unique_counter += 1
#         return self.corr >= 0.8 or len(self.components) >= self.max_components
    

# def main():
#     env = CircuitEnv(max_components=15, value_buckets=5)

#     # Let's define a simple circuit:
#     # Component type → 0: Resistor, 1: Inductor, 2: Capacitor
#     # v_idx is index in value bucket (0 to value_buckets-1)
#     # Nodes n1 and n2 can be arbitrary, let’s use 0, 1, 2 etc.

#     # Add a resistor between node 0 and 1
#     env.step([0, 5, 0, 1])
#     # print(env.components)

#     # Add an inductor between node 1 and 2
#     env.step([1, 3, 1, 1])
#     # print(env.components)

#     # Add a capacitor between node 2 and 0
#     env.step([2, 7, 1, 2])
#     # print(env.components)

#     # Print components added
#     print("Components (ctype, v_idx, n1, n2):")
#     for comp in env.components:
#         print(comp)
#     print(env.get_observation())
#     # print(len(env.components))


# if __name__ == "__main__":
#     main()




# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 17:18:42 2025

@author: Bijeet Basak, Lokesh Kumar, Yash Sengupta
"""

import gymnasium as gym  # The base class for any RL environment
from gymnasium import spaces  # Helps us in defining action and observation spaces
import numpy as np
import networkx as nx  # A graph library to model our circuits as graphs
import read_csv_and_feature_points as rd
import subprocess
import os


simulation_counter = 0
unique_counter = 0
good_circuits = []

uniquecorr = set()

class CircuitEnv(gym.Env):  # Custom Environment


    def __init__(self, max_components=15, value_buckets=5):
        
        # print("Init")
        super(CircuitEnv, self).__init__()

        self.max_components = max_components  # Maximum number of components possible in a circuit.
        self.value_buckets = value_buckets  # Maximum number of possible values each compomnent can have. 
        self.node_counter = 2  # 0 = GND, 1 = VDD --> Each circuit starts with 2 nodes (GND and VDD)
        self.episode_number = 0
        self.freq_points = 10
        self.corr = 0
        
        # You can define custom buckets later for R, L, C
        self.bucket_ranges = {
            0: np.array([1000, 2000, 3000, 4000, 5000]),     # R: 1Ω to 1MΩ
            2: np.array([0.1, 0.5, 1.0, 2.0, 2.5]),   # L: 1uH to 1mH
            1: np.array([3*1e-6, 2*1e-6, 1e-6, 4*1e-6, 5*1e-6])   # C: 1pF to 1uF
        }
        
        self.component_list = ["res", "cap", "ind"]

        # Action = [component_type, value_index, node1, node2]
        # 0 --> R, 1 --> L, 2 --> C  component_type
        # Each value_bucket contains a list of values for each component. The action space will specify the index of the value_buckets[]. Based on the index we will take the value from the bucket.
        # The next two vlaues are the node indices (possible values where we can put a component)
        self.action_space = spaces.MultiDiscrete([3, value_buckets, max_components + 2, max_components + 2])

        # Observation: padded list of components: [type_id, value_idx, node1, node2]
        self.observation_space = spaces.Dict({
            "components": spaces.Box(-1, 1, shape=(self.max_components, 4), dtype=np.float32),
            "target_response": spaces.Box(0, 1, shape=(2, 10), dtype=np.float32),
            })
        
        # Creating the train data object
        folder_path = "train_graphs/csvs"
        folder_path_test = "test_graphs/"
        self.processor = rd.CSVProcessor(folder_path_test)
        
        self.LTSpice = r"C:\Users\lokes\Desktop\LTspice\LTspice.exe"

        self.reset()
        # print("Init Done")
    
    def update_Graph(self, circuit_graph, new_node_value, old_node_value):
        
        # print(f"Components before updaing Graph:{self.components}")
        # self.components = []
        # print(f"New_node={new_node_value}, Old_node={old_node_value}")
        new_components = []
        node_value = None
        for item in self.components:
            if item[3] == old_node_value:
                node_value = new_node_value
            else:
                node_value = item[3]
            new_components.append((item[0], item[1], item[2], node_value))
        self.components = new_components
        new_graph = nx.DiGraph()
        new_graph.add_nodes_from([0, 1])
        for u, v, data in self.G.edges(data=True):
            edge_type = data['type']
            edge_value = data['value']
            # print(f"{edge_value}")
            # print(f"u={u}, v={v}")
            if v == old_node_value:
                # print("v update")
                v = new_node_value
                # print(u, v)
                
            new_graph.add_edge(u, v,  type=edge_type, value=edge_value)
            # self.components.append((edge_type, edge_value, u, v))
            # print(f"Components after updaing Graph:{self.components}")
                
            # Handle node assignment
            for node in [u, v]:
                if node not in new_graph.nodes:
                    new_graph.add_node(node)  # Add the node to the graph representing the circuit
        return new_graph
                
    # LT Spice Interfacing Code
    def build_circuit(self, component_list, filename="generated_circuit.asc"):
        print("Building Circuit")
        print(f"Component list length: {len(component_list)}")
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
    
    def run_simulation(self, circuit_path, ltspice_path):
        global simulation_counter
        simulation_counter += 1
        print(f"simulation_counter: {simulation_counter}")
        counter_file = "simulation_counter.txt"
        with open(counter_file, "w") as f:
            f.write(str(simulation_counter))
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
        
    
    
    
    def plot_response(self, raw_file_path, output_csv="frequency_response.csv"):
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
    
    # LT Spice Interfacing Code ends...


    def reset(self, *, seed=None, options=None):
        
        global good_circuits
        super().reset(seed=seed)
        
        
        good_circuits = []
        self.components = []  # List of components as tuples (type, value_idx, n1, n2)
        self.G = nx.DiGraph()
        self.G.add_nodes_from([0, 1])  # 0 = GND, 1 = VDD
        self.node_counter = 2
        
        
        self.norm_avgs, self.norm_slopes, self.target = self.processor.process_next()
        self.current_file = self.processor.files[self.processor.index - 1]
        
        self.episode_number += 1
        
        obs = self.get_observation()
        
        return obs, {}

    def step(self, action):  # This method is called everytime the agent takes an action
        
        # print("In Step")
    
        new_node_flag = False # This tells if a new node is added in the circuit
        old_node = None # This keeps track of the old node value which now becomes the new node. Useful for updating the graph of nodes.
        # Action unpacking
        comp_type, val_idx, node1, node2 = action
        
        # Is the action valid in the current state or not?
        # This ensures that that there are no floating nodes in the circuit
        
        valid_nodes = len(self.G.nodes) # This gives me the current number of nodes in the circuit as well as the value of the next node.
        # print(f"Valid Node ={valid_nodes}")
        # You can define: current_node_limit = len(valid_nodes)
        if node1 > valid_nodes or node2 > (valid_nodes - 1):
            # Invalid node selection
            return self.get_observation(), -5, False, False, {}
        
        # If both nodes are same, that means one extra node is required to be added
        if node1 == node2:
            
            old_node = node1
            node1 = valid_nodes
            new_node_flag = True
            
        # value = self.get_value(comp_type, val_idx)  # Based on the val_idx and the comp_type, extract the value of the component

        # Handle new node assignment
        for node in [node1, node2]:
            if node not in self.G.nodes:
                self.G.add_node(node)  # Add the node to the graph representing the circuit

        # print("Adding")
        # Add component to circuit
        
        
        # Check if new_node is there. If it is we need to update the entire graph
        if new_node_flag == True:
            # print("Updating Graph")
            self.G = self.update_Graph(self.G, node1, old_node)
            
        self.components.append((comp_type, val_idx, node1, node2))
        
        # print("Adding Edge")
        
        self.G.add_edge(node1, node2, type=comp_type, value=val_idx)
        
        
        
        # Running the simulation and getting the result from LT Spice
        components = []
        gain = []
        # print(self.bucket_ranges)
        
        for item in self.components:
            # print(self.component_list.index(item[0]))
            # print(self.get_value((item[0]), item[1]))
            components.append([self.component_list[item[0]], self.get_value((item[0]), item[1]), item[2], item[3]])
            
        isthere2 = False
        node = list(self.G.nodes)
        for i in node:
            if i == 2:
                isthere2 = True
                break
        for i in range(len(components)):
            if components[i][2] == 2 or components[i][3] == 2:
                isthere2 = True
                break

        if len(components)>2 and isthere2 == True:
            print(isthere2)
            print("isthere2")
            # print(components)
            self.build_circuit(components, "custom_rl_circuit.asc")
            self.run_simulation("custom_rl_circuit.asc", self.LTSpice)
            gain = self.plot_response("custom_rl_circuit.raw", "target.csv")
        

        obs = self.get_observation()
        reward = self.get_reward(gain, self.target)
        done = self.is_done()

        return obs, reward, done, False, {}

    def get_observation(self):
        obs = np.full((self.max_components, 4), -1, dtype=np.float32)

        for i, (ctype, v_idx, n1, n2) in enumerate(self.components[-self.max_components:]):
            value = self.get_value(ctype, v_idx)  # Based on the val_idx and the comp_type, extract the value of the component
            obs[i] = [
                ctype / 2,              # Normalize type (0,1,2)
                value / (self.value_buckets - 1),  # Normalize value index
                n1 / (self.max_components + 1),
                n2 / (self.max_components + 1)
            ]
        target = np.stack(
            [self.norm_avgs, self.norm_slopes], axis=0
        ).astype(np.float32)
        return {
            "components": obs,
            "target_response": target
        }

    def get_value(self, ctype, v_idx):
        
        return self.bucket_ranges[ctype][v_idx]

    def get_reward(self, gain, target):
        
        if len(gain) == 0:
            return -1
        gain1 = np.array(gain).reshape(1, -1)  # Reshape to 2D
        target1 = np.array(target).reshape(1, -1)  # Reshape to 2D
        target1 = np.array(target1, dtype=np.float64)
        
        self.corr = np.corrcoef(gain1, target1)[0, 1]
        reward =-len(self.components)
        if self.corr >= 0.8:
            reward = 10
        return reward

    def is_done(self):
        global unique_counter
        global good_circuits
        if(self.corr >= 0.8):
            print("correlation: " + str(self.corr))
            uniquecorr.add(self.corr)
            print(unique_counter)
            good_circuits.append(self.components)
            # remove duplicated in good_circuits
            good_circuits = [list(x) for x in set(tuple(x) for x in good_circuits)]
            print(f"Good circuits: {len(good_circuits)}")
            print("Unique correlation: " + str(len(uniquecorr)))
            wirter = open("good_circuit.txt", "a")
            for i in good_circuits:
                wirter.write(str(i) + "\n")
            wirter.close()
            # self.build_circuit(self.components, "good_circuit.asc")
            unique_counter += 1
        if(len(self.components) >= self.max_components):
            print("Max components reached")
            # print(unique_counter)
            # unique_counter += 1
        return self.corr >= 0.8 or len(self.components) >= self.max_components
    

def main():
    env = CircuitEnv(max_components=15, value_buckets=5)

    # Let's define a simple circuit:
    # Component type → 0: Resistor, 1: Inductor, 2: Capacitor
    # v_idx is index in value bucket (0 to value_buckets-1)
    # Nodes n1 and n2 can be arbitrary, let’s use 0, 1, 2 etc.

    # Add a resistor between node 0 and 1
    env.step([0, 5, 0, 1])
    # print(env.components)

    # Add an inductor between node 1 and 2
    env.step([1, 3, 1, 1])
    # print(env.components)

    # Add a capacitor between node 2 and 0
    env.step([2, 7, 1, 2])
    # print(env.components)

    # Print components added
    print("Components (ctype, v_idx, n1, n2):")
    for comp in env.components:
        print(comp)
    print(env.get_observation())
    # print(len(env.components))


if __name__ == "__main__":
    main()
