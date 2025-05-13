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
    print("Component list:", component_list)
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
    ['ind', 0.1, 2, 1],
   ['res', 4000, 3, 0],
   ['res', 3000, 0, 1],
   ['cap', 1e-06, 4, 3],
   ['res', 1000, 5, 4],
   ['ind', 2.5, 6, 0],
   ['res', 4000, 7, 1],
   ['cap', 4e-06, 5, 3],
   ['cap', 4e-06, 5, 1],
   ['cap', 4.9999999999999996e-06, 8, 5],
   ['cap', 4.9999999999999996e-06, 9, 5],
   ['cap', 4.9999999999999996e-06, 10, 2],
   ['res', 5000, 10, 9]


]

build_circuit(components, "ckt_0G_1Vin.asc")
