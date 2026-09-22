# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
Figures of the submission.

  python make_figures.py --qasm loop64_314t.qasm --amplitudes probabilities_exp1.csv --out figures/

  t_gates_per_layer   number of T gates per CZ layer
  xeb_distribution    empirical distribution of 2^n p against the Porter-Thomas mixture
  xeb_convergence     cumulative linear XEB with its 95% confidence interval
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse, glob, json, math
import csv
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dcs_rankwidth import load_circuit

plt.rcParams.update({"font.size": 9, "figure.dpi": 160, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
EULER = 0.5772156649015329


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)


def structural_figures(a):
    ops, ring = load_circuit(a.qasm); n = len(ring)
    depth = max(op[2] for op in ops)
    L = np.array([op[2] for op in ops if op[0] == "T"]); hist = np.bincount(L, minlength=depth + 1)
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.bar(np.arange(len(hist)), hist, width=.85, color="C3")
    ax.set_xlabel("CZ layer"); ax.set_ylabel("T gates"); ax.set_xlim(-1, depth + 1)
    ax.annotate(f"{int((L >= depth - 9).sum())} of {len(L)} T gates\nin the last 10 layers",
                xy=(depth - 4, hist.max() * .75), xytext=(depth * .52, hist.max() * .7),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=.8))
    save(fig, a.out, "t_gates_per_layer")



def statistical_figures(a):
    rows = {}
    for f in sorted(glob.glob(a.amplitudes)):
        if f.endswith(".csv"):
            for r in csv.DictReader(open(f)):
                rows.setdefault(int(r["index"]), dict(two_n_p=float(r["two_n_p"])))
        else:
            for l in open(f):
                if l.strip():
                    r = json.loads(l); rows.setdefault(r["i"], r)
    z = np.array([rows[i]["two_n_p"] for i in sorted(rows)]); M = len(z)
    Fs = np.linspace(0, 1, 4001)
    ll = np.array([np.sum(np.log(F * z * np.exp(-z) + (1 - F) * np.exp(-z))) for F in Fs])
    k = int(np.argmax(ll)); F = Fs[k]; ci = Fs[ll >= ll[k] - 1.92]
    cdf = lambda F_, x: F_ * (1 - np.exp(-x) * (1 + x)) + (1 - F_) * (1 - np.exp(-x))

    zs = np.sort(z); emp = np.arange(1, M + 1) / M; xx = np.linspace(0, zs.max(), 400)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax[0].step(zs, emp, where="post", lw=1.4, label=f"data ({M} samples)")
    ax[0].plot(xx, cdf(F, xx), "--", lw=1.2, label=f"mixture, F={F:.3f}")
    ax[0].plot(xx, cdf(0, xx), ":", lw=1.2, color="0.4", label="F=0 (pure noise)")
    ax[0].set_xlabel(r"$2^n p(x)$"); ax[0].set_ylabel("cumulative distribution"); ax[0].set_xlim(0, 8)
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].step(zs, emp - cdf(F, zs), where="post", lw=1.2, label="residual, fitted F")
    ax[1].step(zs, emp - cdf(0, zs), where="post", lw=1.2, color="0.4", ls=":", label="residual, F=0")
    D = 1.36 / math.sqrt(M)
    for s in (1, -1):
        ax[1].axhline(s * D, color="C3", lw=.8, ls="--")
    ax[1].text(5.2, D + .008, "KS 5% threshold", color="C3", fontsize=7)
    ax[1].set_xlabel(r"$2^n p(x)$"); ax[1].set_ylabel("deviation from model"); ax[1].set_xlim(0, 8)
    ax[1].legend(frameon=False, fontsize=8)
    save(fig, a.out, "xeb_distribution")

    n_ = np.arange(1, M + 1); cum = np.cumsum(z - 1) / n_
    sd = np.array([(z[:i] - 1).std(ddof=1) if i > 1 else 0 for i in n_])
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.fill_between(n_, cum - 1.96 * sd / np.sqrt(n_), cum + 1.96 * sd / np.sqrt(n_), alpha=.2, label="95% CI")
    ax.plot(n_, cum, lw=1.3, label="cumulative XEB")
    ax.axhline(.349, color="C3", ls="--", lw=1, label="IBM bound 0.349")
    ax.axhline(.38, color="C2", ls=":", lw=1, label="IBM DFE 0.38")
    ax.set_xlabel("samples"); ax.set_ylabel("linear XEB"); ax.set_ylim(-.1, .9)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, a.out, "xeb_convergence")

    print(f"{M} samples: linear XEB={np.mean(z - 1):.4f}  ML fidelity={F:.4f} [{ci.min():.4f}, {ci.max():.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasm")
    ap.add_argument("--amplitudes"); ap.add_argument("--out", default="figures")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.qasm:
        structural_figures(a)
    if a.amplitudes:
        statistical_figures(a)


if __name__ == "__main__":
    main()
