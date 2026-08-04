#!/usr/bin/env python3
"""Reconstruct coverage_bcp_main.png from the local coverage JSONs.
compute_coverage.py produces coverage_results.json (single-agent pool + 0710 swarm)
and coverage_0711.json (unrestricted-BBS / -isolation swarm). This script only plots;
it reads no cluster paths. Bands are +/-1 SEM; swarm points use N with >=40 supporting cases.
"""
import json, os, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
IMG = os.path.abspath(os.path.join(HERE, "..", "..", "..", "images"))
cr = json.load(open(os.path.join(HERE, "coverage_results.json")))
c11 = json.load(open(os.path.join(HERE, "coverage_0711.json")))

def series(d, min_count=1):
    ks = sorted(int(k) for k in d)
    xs, ys, sem = [], [], []
    for k in ks:
        mean, std, n = d[str(k)]
        if n < min_count:
            continue
        xs.append(k); ys.append(mean); sem.append(std / math.sqrt(n))
    return xs, ys, sem

# pool: all 830 questions at every N; swarm curves: require >=40 supporting cases
pool_x, pool_y, pool_e = series(cr["single_k"])
sw_x, sw_y, sw_e = series(cr["AS_k"], min_count=40)
iso_x, iso_y, iso_e = series(c11["AS_k"], min_count=40)

fig, ax = plt.subplots(figsize=(5.0, 3.6))
def band(x, y, e, style, color, label, ms=5):
    import numpy as np
    x, y, e = np.array(x, float), np.array(y, float), np.array(e, float)
    ax.plot(x, y, style, ms=ms, color=color, label=label)
    ax.fill_between(x, y - e, y + e, color=color, alpha=.15)

# plain-text labels: matplotlib is not in usetex mode, so no \textbf / \% markup
band(sw_x, sw_y, sw_e, "-o", "#1b6ca8", "ArcticSwarm (gated isolation, 82.6%)")
band(iso_x, iso_y, iso_e, "-^", "#2a9d8f", "$-$ gated isolation / free\ncommunication (80.0%)")
band(pool_x, pool_y, pool_e, "--s", "#c1440e", "Independent single-agent\nruns (cumulative)")

ax.set_xlabel("$N$ search paths (browsing agents / runs)")
ax.set_ylabel("Distinct passages retrieved\n(cumulative, mean per question)")
ax.set_xticks(range(0, 21, 4))
ax.grid(alpha=.25)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
out = f"{IMG}/coverage_bcp_main.png"
plt.savefig(out, dpi=220, bbox_inches="tight"); plt.close()
print("wrote", out)
print(f"swarm N1={sw_y[0]:.0f} N10={sw_y[9]:.0f} (Nmax={sw_x[-1]}); "
      f"pool N1={pool_y[0]:.0f} N10={pool_y[9]:.0f}; iso N10={iso_y[9]:.0f}")
