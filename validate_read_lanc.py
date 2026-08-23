"""Compares the OLD (nested-loop) and NEW (vectorized) local-ancestry-file
parsing logic on synthetic data, for both the space-delimited and compact
single-digit file formats, and checks the resulting lanc_dict structures are
identical.
"""
import numpy as np
import time
import random


def make_ancestry_file(path, n_snps, n_individuals, n_ancs, space_delimited, seed=0):
    r = random.Random(seed)
    lines = []
    for _ in range(n_snps):
        labels = [r.randint(1, n_ancs) for _ in range(2 * n_individuals)]
        if space_delimited:
            lines.append(" ".join(str(x) for x in labels))
        else:
            lines.append("".join(str(x) for x in labels))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def old_read_lanc(lanc_file, no_of_ancs, lanc_labels_list):
    """Verbatim port of the ORIGINAL nested-loop algorithm."""
    lanc_dict = {}
    lineno = 0
    with open(lanc_file) as fh:
        for line in fh:
            lineno += 1
            data = line.rstrip().split(" ")
            if len(data[0]) > 1:
                data = list(line.rstrip())
            lanc_dict[lineno] = {}
            for idx in range(no_of_ancs):
                lanc_dict[lineno][idx + 1] = []
            for hap in range(0, len(data), 2):
                if data[hap] == data[hap + 1]:
                    for anc in lanc_labels_list:
                        if int(data[hap]) == anc:
                            lanc_dict[lineno][anc].append(2)
                        else:
                            lanc_dict[lineno][anc].append(0)
                else:
                    for anc2 in lanc_labels_list:
                        if int(data[hap]) == anc2 or int(data[hap + 1]) == anc2:
                            lanc_dict[lineno][anc2].append(1)
                        else:
                            lanc_dict[lineno][anc2].append(0)
    return lanc_dict


def new_read_lanc(lanc_file, no_of_ancs, lanc_labels_list):
    """The optimized/vectorized version (logic mirrors JasMAP4.py's read_lanc)."""
    with open(lanc_file) as fh:
        raw_lines = [l.rstrip('\n').rstrip('\r') for l in fh]
    raw_lines = [l for l in raw_lines if l != '']

    space_delimited = (' ' in raw_lines[0])
    if space_delimited:
        hap_matrix = np.array([line.split() for line in raw_lines], dtype=np.int8)
    else:
        n_hap = len(raw_lines[0])
        hap_matrix = np.empty((len(raw_lines), n_hap), dtype=np.int8)
        for i, line in enumerate(raw_lines):
            hap_matrix[i, :] = np.frombuffer(line.encode('ascii'), dtype=np.uint8) - ord('0')

    hap0 = hap_matrix[:, 0::2]
    hap1 = hap_matrix[:, 1::2]
    lanc_array = np.stack(
        [((hap0 == anc).astype(np.int8) + (hap1 == anc).astype(np.int8)) for anc in lanc_labels_list],
        axis=0)

    n_snps = hap_matrix.shape[0]
    lanc_dict = {}
    per_anc_lists = [lanc_array[k].tolist() for k in range(no_of_ancs)]
    for snp_idx in range(n_snps):
        lineno = snp_idx + 1
        lanc_dict[lineno] = {anc: per_anc_lists[anc - 1][snp_idx] for anc in lanc_labels_list}
    return lanc_dict, lanc_array


if __name__ == "__main__":
    import tempfile, os
    all_ok = True
    for space_delimited in (True, False):
        for (n_snps, n_ind, n_ancs) in [(20, 5, 2), (200, 30, 3)]:
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, "anc.txt")
                make_ancestry_file(path, n_snps, n_ind, n_ancs, space_delimited, seed=n_snps + n_ind)
                lanc_labels_list = [i + 1 for i in range(n_ancs)]

                t0 = time.time()
                old_dict = old_read_lanc(path, n_ancs, lanc_labels_list)
                t_old = time.time() - t0

                t0 = time.time()
                new_dict, lanc_array = new_read_lanc(path, n_ancs, lanc_labels_list)
                t_new = time.time() - t0

                match = (old_dict == new_dict)
                all_ok &= match
                fmt = "space" if space_delimited else "compact"
                print(f"[{fmt:7s}] n_snps={n_snps:4d} n_ind={n_ind:3d} n_ancs={n_ancs}  "
                      f"MATCH={match}  old={t_old*1000:7.2f}ms  new={t_new*1000:7.2f}ms  "
                      f"speedup={t_old/max(t_new,1e-9):6.1f}x")

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
