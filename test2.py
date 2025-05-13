# test_with_clamp.py

import time, subprocess
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from CustomEnv import CircuitEnv

def load_target_response(path):
    df = pd.read_csv(path,
        converters={'Frequency': lambda x: complex(x).real,
                    'Gain': float})
    freq = df['Frequency'].values
    gain = df['Gain'].values
    norm_gain = gain / gain.max()
    logf = np.log10(freq)
    slope = np.gradient(norm_gain, logf)
    smin, smax = slope.min(), slope.max()
    norm_slope = (slope - smin) / (smax - smin + 1e-8)
    return np.vstack([norm_gain, norm_slope])

# 1) Patch run_simulation for a 60s timeout & log
def patched_run_simulation(self, circuit_path, ltspice_path):
    print(f"[run_sim] Launching LTspice on {circuit_path}")
    try:
        res = subprocess.run(
            [ltspice_path, "-b", "-run", "-ascii", circuit_path],
            check=True, capture_output=True, text=True, timeout=60
        )
        print(f"[run_sim] Done in {res}")
    except subprocess.TimeoutExpired:
        print("[run_sim] ERROR: timed out")
    except subprocess.CalledProcessError as e:
        print(f"[run_sim] ERROR rc={e.returncode}\n{e.stderr}")
CircuitEnv.run_simulation = patched_run_simulation

# 2) Monkey‐patch step() to clamp invalid node indices
_orig_step = CircuitEnv.step
def clamped_step(self, action):
    comp, val, n1, n2 = action
    valid = len(self.G.nodes)  # valid nodes are 0..valid

    fixed = [int(comp), int(val), n1, n2]
    if fixed != list(action):
        print(f"  • clamped {action} → {fixed}")
    return _orig_step(self, fixed)
CircuitEnv.step = clamped_step

# 3) Agent runner & ASC builder
def run_and_build(env, model, outfile="out_circuit.asc"):
    obs, _ = env.reset()
    done, steps = False, 0
    while not done and steps < 10000:
        raw, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(raw)
        print(f"Step {steps:02d}: raw={raw}, reward={reward:.1f}, done={done}")
        steps += 1

    # write out .asc
    comps = [(env.component_list[c], env.get_value(c, v), n1, n2)
             for (c, v, n1, n2) in env.components]
    env.build_circuit(comps, filename=outfile)
    print(f"Wrote schematic → {outfile}")
    return comps

def main():
    model = PPO.load("ppo_best_model/best_model.zip")
    env   = CircuitEnv(max_components=15, value_buckets=5)

    target = load_target_response("my_target.csv")
    env.norm_avgs, env.norm_slopes, env.target = target[0], target[1], target

    # reset once more to apply override cleanly
    obs, _ = env.reset()
    env.norm_avgs, env.norm_slopes, env.target = target[0], target[1], target

    t0 = time.time()
    comps = run_and_build(env, model)
    print(f"Done in {time.time()-t0:.2f}s")
    print("Final components:")
    for c in comps:
        print("  ", c)

if __name__ == "__main__":
    main()
