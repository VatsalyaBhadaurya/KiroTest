"""
Quick profiler — measures cold-start and warm inference latency.
Run: python profile_inference.py
"""

import time
import numpy as np
from PIL import Image

# Synthetic 224x224 RGB image
dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

print("=" * 50)
print("VLA Agent — Inference Profiler")
print("=" * 50)

# Cold start (model load + inference)
t0 = time.perf_counter()
from vlm_reasoner import reason
from action_generator import generate_action

scene, ms = reason(dummy, "pick up the bottle")
action = generate_action(scene, "pick up the bottle")
cold_ms = (time.perf_counter() - t0) * 1000

print(f"\nCold start (model load + inference): {cold_ms:.0f} ms")
print(f"  reasoning_ms : {ms:.1f}")
print(f"  action       : {action}")

# Warm inference (model already loaded)
runs = 5
times = []
for _ in range(runs):
    t = time.perf_counter()
    scene, ms = reason(dummy)
    generate_action(scene)
    times.append((time.perf_counter() - t) * 1000)

print(f"\nWarm inference ({runs} runs):")
print(f"  avg : {np.mean(times):.1f} ms")
print(f"  min : {np.min(times):.1f} ms")
print(f"  max : {np.max(times):.1f} ms")
print("=" * 50)
