"""Basic backstepping helper example."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from cbf_safety_box.safety_data import SafetySample
from cbf_safety_box.constraints.backstepping import auxiliary_k1, compute_backstepping_value

out = Path("outputs/backstepping_basic")
out.mkdir(parents=True, exist_ok=True)
safety = SafetySample(h=1.0, grad_h=np.array([1.0, 0.0]))
k1 = auxiliary_k1(safety, "gradient_ascent", gain=1.0)
vels = np.linspace(-3, 3, 200)
hB = []
for vx in vels:
    hB.append(compute_backstepping_value(safety, np.array([vx, 0.0]), mu=1.0, k1=k1)["h_B"])
plt.figure(figsize=(7,4))
plt.plot(vels, hB)
plt.axhline(0, color="k", linewidth=1)
plt.title("Backstepping candidate h_B vs velocity")
plt.xlabel("v_x")
plt.ylabel("h_B")
plt.grid(True)
plt.savefig(out / "backstepping_hB.png", dpi=180, bbox_inches="tight")
print("Saved", out / "backstepping_hB.png")
