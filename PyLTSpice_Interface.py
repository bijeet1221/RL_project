def generate_simple_asc(filename="simple_circuit.asc"):
    lines = [
        "Version 4",
        "SHEET 1 880 680",

        # --- Wires ---
        "WIRE 160 96 112 96",   # V1 top wire
        "WIRE 160 96 160 144",  # V1 to R1
        # "WIRE 160 192 160 240", # R1 to C1
        # "WIRE 160 288 160 320", # C1 to GND

        # # --- Voltage Source V1 ---
        # "SYMBOL voltage 112 80 R0",
        # "WINDOW 3 32 56 Left 0",
        # "SYMATTR InstName V1",
        # "SYMATTR Value DC 5",

        # # # --- Resistor R1 ---
        # "SYMBOL res 144 112 R0",
        # "SYMATTR InstName R1",
        # "SYMATTR Value 1k",

        # # --- Capacitor C1 ---
        # "SYMBOL cap 160 240 R0",
        # "SYMATTR InstName C1",
        # "SYMATTR Value 10u",

        # # --- Ground ---
        "FLAG 160 320 0",

        # # --- Simulation command ---
        # "TEXT 80 360 Left 0 !.tran 1m"
    ]

    with open(filename, "w") as f:
        f.write("\n".join(lines))
    print(f"Schematic saved to '{filename}'")

generate_simple_asc()