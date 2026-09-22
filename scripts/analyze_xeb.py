# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
Fidelity estimates from exact output probabilities.

  python analyze_xeb.py "amplitudes_*.jsonl" [--expect 1389]
  python analyze_xeb.py probabilities_exp1.csv --expect 1389        (Zenodo file) [--label name] [--ro-fid F] [--json out]

Reports, with z = 2^n p(x) over the measured bitstrings:
  - linear XEB, mean(z) - 1, and log-XEB, mean(log z) + gamma, with normal and bootstrap 95% CIs;
  - the maximum-likelihood fidelity under the Porter-Thomas mixture
        f(z) = F z e^{-z} + (1 - F) e^{-z},
    with a profile-likelihood 95% CI (delta log L = 1.92);
  - a Kolmogorov-Smirnov test against that mixture;
  - integrity checks: duplicated indices, inconsistent duplicates, missing indices.
--ro-fid divides the estimates by a readout fidelity (see S4.3 of arXiv:2607.25941).
Quote file patterns so that the script, not the shell, expands them.
"""
import argparse, collections, glob, json, math
import numpy as np

EULER = 0.5772156649015329


def read_rows(path):
    """Read one results file: JSON lines (fields i, p, two_n_p) or CSV (columns index, probability,
    two_n_p), such as the files of the Zenodo record."""
    if path.endswith(".csv"):
        import csv
        with open(path) as fh:
            return [dict(i=int(r["index"]), p=float(r["probability"]), two_n_p=float(r["two_n_p"]))
                    for r in csv.DictReader(fh)]
    return [json.loads(l) for l in open(path) if l.strip()]


def load(patterns):
    rows = []
    for p in patterns:
        for f in sorted(glob.glob(p)):
            rows += read_rows(f)
    d = collections.defaultdict(list)
    for r in rows:
        d[r["i"]].append(r)
    inconsistent = [i for i, v in d.items()
                    if len(v) > 1 and max(x["two_n_p"] for x in v) - min(x["two_n_p"] for x in v) > 1e-6]
    return d, len(rows), inconsistent


def ml_fidelity(z):
    Fs = np.linspace(0, 1, 4001)
    ll = np.array([np.sum(np.log(F * z * np.exp(-z) + (1 - F) * np.exp(-z))) for F in Fs])
    k = int(np.argmax(ll)); ci = Fs[ll >= ll[k] - 1.92]
    return Fs[k], ci.min(), ci.max()


def ks_distance(z, F):
    zs = np.sort(z); M = len(zs)
    cdf = F * (1 - np.exp(-zs) * (1 + zs)) + (1 - F) * (1 - np.exp(-zs))
    return float(np.max(np.abs(np.arange(1, M + 1) / M - cdf)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ro-fid", type=float, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default=None)
    ap.add_argument("--expect", type=int, default=None)
    a = ap.parse_args()
    d, nrows, inconsistent = load(a.files)
    z = np.array([d[i][0]["two_n_p"] for i in sorted(d)]); M = len(z)
    print(f"=== {a.label or ' '.join(a.files)}")
    print(f"lines read={nrows}  unique indices={M}  inconsistent duplicates={len(inconsistent)}")
    if a.expect:
        missing = sorted(set(range(a.expect)) - set(d))
        print(f"missing out of {a.expect}: {len(missing)}" + (f" -> {missing[:20]}" if missing else ""))
    lin, lg = z - 1, np.log(z) + EULER
    rng = np.random.default_rng(0); idx = rng.integers(0, M, (20000, M))
    bl = (z[idx] - 1).mean(axis=1); bg = (np.log(z[idx]) + EULER).mean(axis=1)
    F, lo, hi = ml_fidelity(z)
    res = dict(M=M, lin=lin.mean(), lin_ci95=1.96 * lin.std(ddof=1) / math.sqrt(M),
               lin_boot=list(np.percentile(bl, [2.5, 97.5])), log=lg.mean(),
               log_ci95=1.96 * lg.std(ddof=1) / math.sqrt(M), log_boot=list(np.percentile(bg, [2.5, 97.5])),
               F_ml=F, F_ml_ci=[lo, hi], ks=ks_distance(z, F), ks_threshold=1.36 / math.sqrt(M))
    print(f"linear XEB : {res['lin']:.4f} +- {res['lin_ci95']:.4f}   bootstrap [{res['lin_boot'][0]:.4f}, {res['lin_boot'][1]:.4f}]")
    print(f"log-XEB    : {res['log']:.4f} +- {res['log_ci95']:.4f}   bootstrap [{res['log_boot'][0]:.4f}, {res['log_boot'][1]:.4f}]")
    print(f"ML fidelity: {F:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"KS vs fitted mixture: D={res['ks']:.4f}   5% threshold={res['ks_threshold']:.4f}"
          f"   -> {'compatible' if res['ks'] < res['ks_threshold'] else 'INCOMPATIBLE'}")
    for f0 in (0.0, 0.349, 0.38):
        print(f"   KS vs F={f0}: D={ks_distance(z, f0):.4f}")
    if a.ro_fid:
        print(f"readout-corrected (F_ro={a.ro_fid}): XEB={res['lin'] / a.ro_fid:.4f}  F_ml={F / a.ro_fid:.4f}")
        res.update(ro_fid=a.ro_fid, lin_corrected=res["lin"] / a.ro_fid, F_ml_corrected=F / a.ro_fid)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
