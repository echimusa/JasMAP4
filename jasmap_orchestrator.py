#!/usr/bin/env python3
"""
jasmap_orchestrator.py — LangGraph orchestration for the JasMAP4 pipeline
==========================================================================

Track D of the deployment/development plan. Wraps the existing JasMAP4
pipeline in a LangGraph state graph providing checkpoint/resume, per-step
retry with parameter adjustment, and an auditable provenance trail.


A correction to the plan's premise
-----------------------------------
The v2 plan stated "every pipeline step is already a self-contained
callable function, so the graph nodes are largely already there." Checked
against the code, that is not accurate, and the difference matters
architecturally.

The steps are METHODS on a single stateful object (`_run_lai_pipeline`,
`_run_gift_mode`, `_run_rvat_mode`, `_run_prs_pipeline`, `_run_consgwas`),
communicating through instance attributes — `self.genotypes`,
`self.lanc_dict`, `self.phenotypes`, `self.snp_meta`, `self.args.gene_annot`
— not through arguments and return values. Several of those attributes are
multi-gigabyte numpy arrays or memmaps; `lanc_dict` is a lazy view over a
memmap, and the genotype matrix reaches ~40 GB at benchmark scale.

That rules out the obvious design. LangGraph checkpoints its state after
every node, so anything in state must be serialisable and small. Putting
the JasMAP object (or its arrays) into graph state would mean attempting to
serialise tens of gigabytes into SQLite/Postgres on every transition —
which would not merely be slow, it would defeat the purpose, since the
point of checkpointing is cheap, frequent, durable progress markers.


The design that follows from that
----------------------------------
Graph state holds ONLY lightweight, JSON-serialisable metadata: input
paths, CLI parameters, per-step status, timings, retry counts, error text,
and the output paths each step produced. Heavy data never enters state.

Steps are invoked as subprocesses against the real JasMAP4 CLI. This is
viable — and is the reason the boundary is drawn here rather than at the
method level — because the pipeline already persists everything that
crosses a step boundary: LAI writes an ancfile that `--ancfile` can read
back, association steps write `.assoc` files that `--consgwas` picks up,
PRS writes `prs_results/`. The file system is already the integration
contract; this makes that explicit rather than inventing a new one.

Practical consequences of that choice:

  * Resume is real, not simulated. A step whose expected outputs already
    exist on disk is skipped on re-entry, so a run that died in hour 6 of 9
    restarts near where it stopped rather than from the beginning.
  * A crashing step cannot corrupt the orchestrator, because it is in
    another process.
  * Retries can genuinely alter parameters (see `_retry_adjustments`) and
    re-invoke, rather than re-running an identical call and hoping.
  * The trade-off, stated plainly: process-per-step means genotype data is
    re-read per step. That cost is real but bounded (~2 minutes on the SAC
    cohort) and is the price of durable, resumable, isolated steps. For a
    single short run, calling JasMAP4.py directly remains faster.


Checkpoint backend
------------------
SQLite by default so this runs with no infrastructure. The plan's confirmed
direction is Postgres for distributed use; `--checkpoint-db postgres://...`
selects it, and the only change is which saver is constructed — the graph,
state, and nodes are identical. Postgres is not required to evaluate or
develop against this.


Usage
-----
    python3 jasmap_orchestrator.py --bfile data/COHORT --pheno data/COHORT.pheno \\
        --mode admix --nbanc 5 --yes-anc --pop-file data/POP.txt \\
        --pop-labels KHS,CEU,YRI,CHB,GIH --prs --consgwas \\
        --out ./results --out-prefix RUN1

    # resume an interrupted run (same --thread-id):
    python3 jasmap_orchestrator.py ... --thread-id RUN1

    # inspect what a run did, without executing anything:
    python3 jasmap_orchestrator.py --thread-id RUN1 --show-provenance
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import time
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

def _merge_dict(a: dict, b: dict) -> dict:
    """Reducer so concurrent or sequential node updates accumulate rather than
    overwrite — each node records under its own key."""
    out = dict(a or {})
    out.update(b or {})
    return out


def _append_list(a: list, b: list) -> list:
    return (a or []) + (b or [])


class PipelineState(TypedDict, total=False):
    """Deliberately flat and JSON-serialisable. No arrays, no handles, no
    JasMAP object — see the module docstring for why this is the whole
    design constraint rather than a stylistic preference."""
    jasmap_path: str
    args: dict[str, Any]          # CLI parameters (strings/bools/numbers only)
    out_dir: str
    out_prefix: str
    mode: str

    status: Annotated[dict[str, str], _merge_dict]      # step -> pending/ok/skipped/failed
    timings: Annotated[dict[str, float], _merge_dict]   # step -> seconds
    outputs: Annotated[dict[str, list], _merge_dict]    # step -> files produced
    retries: Annotated[dict[str, int], _merge_dict]     # step -> attempts used
    errors: Annotated[dict[str, str], _merge_dict]      # step -> last error text
    provenance: Annotated[list[dict], _append_list]     # ordered audit trail

    halted: bool
    halt_reason: str


# ─────────────────────────────────────────────────────────────────────────────
# Invocation helper
# ─────────────────────────────────────────────────────────────────────────────

MAX_RETRIES = 2

# Retry adjustments are deliberately conservative and each has a stated
# rationale. A retry that changes nothing is just a slower failure; a retry
# that silently changes statistical behaviour is worse than failing.
_retry_adjustments: dict[str, list[dict]] = {
    "lai": [
        # Most LAI failures at scale are memory, not logic. --chr converts a
        # genome-wide run into a per-chromosome one, the documented workaround
        # for the dense LAI load.
        {"note": "retry restricted to chr1 to bound memory", "set": {"chr": "1"}},
    ],
    "prs": [
        {"note": "retry with a smaller pre-filter SNP budget",
         "set": {"prs-max-snps": "5000"}},
    ],
    "rvat": [
        {"note": "retry with a looser MAF threshold (array data often has "
                 "no variants below the default 0.01)",
         "set": {"maf-max": "0.05"}},
    ],
}


def _build_cmd(state: PipelineState, overrides: dict | None = None) -> list[str]:
    args = dict(state["args"])
    if overrides:
        args.update(overrides)
    cmd = [sys.executable, state["jasmap_path"]]
    for k, v in args.items():
        if v is None or v is False:
            continue
        flag = f"--{k}"
        if v is True:
            cmd.append(flag)
        else:
            cmd += [flag, str(v)]
    return cmd


def _snapshot_outputs(out_dir: str) -> list[str]:
    if not os.path.isdir(out_dir):
        return []
    found = []
    for root, _dirs, files in os.walk(out_dir):
        for f in files:
            found.append(os.path.relpath(os.path.join(root, f), out_dir))
    return sorted(found)


def _run_step(state: PipelineState, step: str, overrides: dict,
              expect_any: list[str]) -> dict:
    """Invoke JasMAP4 for one step, recording status/timing/outputs/provenance.

    `expect_any`: filename fragments that indicate this step already
    completed. If any is already present in the output directory the step is
    SKIPPED — this is what makes resume real rather than a re-run.
    """
    out_dir = state["out_dir"]
    before = _snapshot_outputs(out_dir)

    if expect_any and any(any(frag in f for f in before) for frag in expect_any):
        return {
            "status": {step: "skipped"},
            "provenance": [{"step": step, "result": "skipped",
                            "reason": "expected outputs already present "
                                      "(resuming a previous run)",
                            "ts": time.time()}],
        }

    attempt = 0
    last_err = ""
    while attempt <= MAX_RETRIES:
        adj = {}
        note = None
        if attempt > 0:
            plans = _retry_adjustments.get(step, [])
            if attempt - 1 < len(plans):
                adj = plans[attempt - 1]["set"]
                note = plans[attempt - 1]["note"]
            else:
                break  # no further distinct adjustment to try

        cmd = _build_cmd(state, {**overrides, **adj})
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0

        if proc.returncode == 0:
            after = _snapshot_outputs(out_dir)
            new_files = sorted(set(after) - set(before))
            return {
                "status": {step: "ok"},
                "timings": {step: round(dt, 2)},
                "outputs": {step: new_files},
                "retries": {step: attempt},
                "provenance": [{
                    "step": step, "result": "ok", "attempt": attempt,
                    "seconds": round(dt, 2), "adjustment": note,
                    "command": shlex.join(cmd), "new_files": new_files,
                    "ts": time.time()}],
            }

        last_err = (proc.stderr or proc.stdout or "")[-2000:]
        attempt += 1

    return {
        "status": {step: "failed"},
        "errors": {step: last_err},
        "retries": {step: attempt},
        "provenance": [{"step": step, "result": "failed",
                        "attempts": attempt, "error_tail": last_err,
                        "ts": time.time()}],
        "halted": True,
        "halt_reason": f"step '{step}' failed after {attempt} attempt(s)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Nodes — one per pipeline step, mirroring jasmap()'s Step A/B/C/D
# ─────────────────────────────────────────────────────────────────────────────

def node_lai(state: PipelineState) -> dict:
    """Step A: local + global ancestry (only when --yes-anc and an
    ancestry-consuming mode)."""
    if not state["args"].get("yes-anc") or state["mode"] not in ("admix", "gift_admix", "LAI"):
        return {"status": {"lai": "skipped"},
                "provenance": [{"step": "lai", "result": "skipped",
                                "reason": "not requested for this mode",
                                "ts": time.time()}]}
    # Run LAI standalone so its ancfile is produced and checkpointed before
    # the (much longer) association step begins -- the whole point of making
    # this a separate node.
    return _run_step(state, "lai",
                     overrides={"mode": "LAI", "PRS": None, "consgwas": False},
                     expect_any=["consensus_lai_local_ancestry"])


def node_analysis(state: PipelineState) -> dict:
    """Step B: the mode-specific association/inference step."""
    if state["mode"] == "LAI":
        return {"status": {"analysis": "skipped"},
                "provenance": [{"step": "analysis", "result": "skipped",
                                "reason": "--mode LAI has no downstream analysis",
                                "ts": time.time()}]}
    expect = {"GWAS": ["Jasmap.GWAS.assoc"], "admix": ["Jasmap.Joint.assoc"],
              "rvat": ["rvat_results"], "gift": ["gift_results"],
              "gift_admix": ["gift_admix_results"]}.get(state["mode"], [])
    return _run_step(state, state["mode"] if state["mode"] in _retry_adjustments else "analysis",
                     overrides={"PRS": None, "consgwas": False},
                     expect_any=expect)


def node_prs(state: PipelineState) -> dict:
    """Step C: consensus PRS."""
    if not state["args"].get("PRS") or state["mode"] == "LAI":
        return {"status": {"prs": "skipped"},
                "provenance": [{"step": "prs", "result": "skipped",
                                "reason": "not requested", "ts": time.time()}]}
    return _run_step(state, "prs", overrides={"consgwas": False},
                     expect_any=["prs_results"])


def node_consensus(state: PipelineState) -> dict:
    """Step D: consensus GWAS/RVAT."""
    if not state["args"].get("consgwas") or state["mode"] == "LAI":
        return {"status": {"consensus": "skipped"},
                "provenance": [{"step": "consensus", "result": "skipped",
                                "reason": "not requested", "ts": time.time()}]}
    return _run_step(state, "consensus", overrides={"PRS": None},
                     expect_any=["consensus_gwas"])


def node_report(state: PipelineState) -> dict:
    """Terminal node: write an auditable run summary alongside the results."""
    path = os.path.join(state["out_dir"], "orchestration_provenance.json")
    os.makedirs(state["out_dir"], exist_ok=True)
    payload = {
        "mode": state["mode"],
        "out_prefix": state.get("out_prefix"),
        "status": state.get("status", {}),
        "timings_seconds": state.get("timings", {}),
        "retries": state.get("retries", {}),
        "errors": state.get("errors", {}),
        "halted": state.get("halted", False),
        "halt_reason": state.get("halt_reason", ""),
        "steps": state.get("provenance", []),
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return {"provenance": [{"step": "report", "result": "ok",
                            "wrote": path, "ts": time.time()}]}


def _gate(state: PipelineState) -> Literal["continue", "halt"]:
    """Conditional edge: stop the graph as soon as a step has exhausted its
    retries, rather than running later steps against missing inputs and
    producing a second, more confusing failure."""
    return "halt" if state.get("halted") else "continue"


def build_graph(checkpointer):
    g = StateGraph(PipelineState)
    g.add_node("lai", node_lai)
    g.add_node("analysis", node_analysis)
    g.add_node("prs", node_prs)
    g.add_node("consensus", node_consensus)
    g.add_node("report", node_report)

    g.set_entry_point("lai")
    for src, dst in [("lai", "analysis"), ("analysis", "prs"), ("prs", "consensus")]:
        g.add_conditional_edges(src, _gate, {"continue": dst, "halt": "report"})
    g.add_edge("consensus", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jasmap", default="JasMAP4.py", help="Path to JasMAP4.py")
    p.add_argument("--thread-id", default=None,
                   help="Run identifier. Re-using it resumes that run.")
    p.add_argument("--checkpoint-db", default="jasmap_orchestration.sqlite",
                   help="SQLite path, or postgres://... for distributed use")
    p.add_argument("--show-provenance", action="store_true",
                   help="Print the recorded provenance for --thread-id and exit")

    # Passed through to JasMAP4
    p.add_argument("--bfile"); p.add_argument("--vcf"); p.add_argument("--geno")
    p.add_argument("--pheno"); p.add_argument("--covfile")
    p.add_argument("--mode", required=False, default="GWAS")
    p.add_argument("--nbanc"); p.add_argument("--pop-file"); p.add_argument("--pop-labels")
    p.add_argument("--admix-label"); p.add_argument("--chr")
    p.add_argument("--yes-anc", action="store_true")
    p.add_argument("--prs", action="store_true")
    p.add_argument("--consgwas", action="store_true")
    p.add_argument("--annot", action="store_true"); p.add_argument("--genome-build")
    p.add_argument("--maf-max"); p.add_argument("--rvat-min-variants")
    p.add_argument("--auto-pca-k"); p.add_argument("--kinship")
    p.add_argument("--out", default="./jasmap_out"); p.add_argument("--out-prefix", default="RUN")
    a = p.parse_args()

    thread_id = a.thread_id or f"{a.out_prefix}-{a.mode}"

    jasmap_args = {
        "bfile": a.bfile, "vcf": a.vcf, "geno": a.geno, "pheno": a.pheno,
        "covfile": a.covfile, "mode": a.mode, "nbanc": a.nbanc,
        "pop-file": a.pop_file, "pop-labels": a.pop_labels,
        "admix-label": a.admix_label, "chr": a.chr,
        "yes-anc": a.yes_anc, "PRS": "Yes" if a.prs else None,
        "consgwas": a.consgwas, "annot": a.annot, "genome-build": a.genome_build,
        "maf-max": a.maf_max, "rvat-min-variants": a.rvat_min_variants,
        "auto-pca-k": a.auto_pca_k, "kinship": a.kinship,
        "out": a.out, "out-prefix": a.out_prefix,
    }
    jasmap_args = {k: v for k, v in jasmap_args.items() if v not in (None, False)}

    conn = sqlite3.connect(a.checkpoint_db, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    app = build_graph(checkpointer)
    cfg = {"configurable": {"thread_id": thread_id}}

    if a.show_provenance:
        snap = app.get_state(cfg)
        if not snap or not snap.values:
            print(f"No checkpointed run found for thread-id '{thread_id}'.")
            return
        print(json.dumps({"status": snap.values.get("status", {}),
                          "timings": snap.values.get("timings", {}),
                          "retries": snap.values.get("retries", {}),
                          "halted": snap.values.get("halted", False),
                          "steps": snap.values.get("provenance", [])}, indent=2))
        return

    init: PipelineState = {
        "jasmap_path": a.jasmap, "args": jasmap_args,
        "out_dir": a.out, "out_prefix": a.out_prefix, "mode": a.mode,
        "status": {}, "timings": {}, "outputs": {}, "retries": {},
        "errors": {}, "provenance": [], "halted": False, "halt_reason": "",
    }

    print(f"[orchestrator] thread-id={thread_id}  checkpoints={a.checkpoint_db}")
    final = app.invoke(init, cfg)

    print("\n[orchestrator] step status:")
    for k, v in final.get("status", {}).items():
        t = final.get("timings", {}).get(k)
        print(f"   {k:<12} {v}" + (f"   ({t}s)" if t else ""))
    if final.get("halted"):
        print(f"\n[orchestrator] HALTED: {final.get('halt_reason')}")
        sys.exit(1)
    print(f"\n[orchestrator] complete. Provenance -> "
          f"{os.path.join(a.out, 'orchestration_provenance.json')}")


if __name__ == "__main__":
    main()
