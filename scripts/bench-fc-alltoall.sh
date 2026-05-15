#!/usr/bin/env bash
# A/B benchmark for flowcontrol-guard-cc on AlltoAll.
#
# Supports two launch modes:
#   local : single-host, nccl-tests multi-threaded (-t $NGPU). Forces network
#           transport via NCCL_P2P_DISABLE/NCCL_SHM_DISABLE — synthetic test,
#           HCA loopback, not representative of real fabric behavior.
#   mpi   : multi-host via mpirun + HOSTFILE. One process per GPU; cross-node
#           traffic goes through the real IB/RoCE fabric. This is the only
#           way to validate FC's congestion-control claims.
#
# Auto-engages mpi mode when HOSTFILE is set. See README block at bottom for
# preconditions and a runbook.

set -euo pipefail

# ============================================================ Config ====
NCCL_ROOT=${NCCL_ROOT:-/home/scratch.jiayaog_gpu/repo/nccl}
NCCL_TESTS_ROOT=${NCCL_TESTS_ROOT:-/home/scratch.jiayaog_gpu/repo/nccl-tests}
NGPU=${NGPU:-8}                 # GPUs per node (single-node) / GPUS_PER_NODE alias
GPUS_PER_NODE=${GPUS_PER_NODE:-$NGPU}
TRIALS=${TRIALS:-3}
SIZE_MIN=${SIZE_MIN:-4K}
SIZE_MAX=${SIZE_MAX:-1G}
ITERS=${ITERS:-30}
WARMUP=${WARMUP:-10}
OUTDIR=${OUTDIR:-$NCCL_ROOT/scripts/fc-bench-results}
QUICK=${QUICK:-0}               # QUICK=1 => 1 trial, 1M..16M only — for smoke test
MODE=${MODE:-all}               # all | baseline | fc-default | fc-guard

# ---- Launcher ---------------------------------------------------------
# LAUNCHER:
#   auto  — pick "mpi" if HOSTFILE is set, else "local" (default)
#   local — single-host nccl-tests multi-threaded
#   mpi   — mpirun + HOSTFILE, one process per GPU
LAUNCHER=${LAUNCHER:-auto}
HOSTFILE=${HOSTFILE:-}                # MPI hostfile path (one host per line: "host slots=N")
MPI_BIN=${MPI_BIN:-mpirun}            # override if mpirun is named differently (mpiexec, srun-mpi…)
MPI_EXTRA_ARGS=${MPI_EXTRA_ARGS:-}    # extra args to forward to mpirun (e.g. "-mca pml ob1")

# ---- NIC topology ----------------------------------------------------
# spread — default; in local mode use 8 NICs 1:1, in mpi mode use all NICs
# pin1   — all ranks pinned to a single HCA, creating real incast on that NIC
NIC_MODE=${NIC_MODE:-spread}
PIN_NIC=${PIN_NIC:-mlx5_0}
# When pinning, FC's topology auto-discovery returns share=1 (it walks the
# PCIe graph, not NCCL_IB_HCA). Set this to override BW per rank manually.
# 0 = leave auto-discovery; integer GB/s = override.
FC_PIN_BW_GBS=${FC_PIN_BW_GBS:-0}

if [[ "$QUICK" == "1" ]]; then
  TRIALS=1; SIZE_MIN=1M; SIZE_MAX=16M; ITERS=10; WARMUP=5
fi

mkdir -p "$OUTDIR"

# ============================================================ Launcher ==
if [[ "$LAUNCHER" == "auto" ]]; then
  if [[ -n "$HOSTFILE" ]]; then LAUNCHER=mpi; else LAUNCHER=local; fi
fi

NODES=1
TOTAL_PROCS=$NGPU
if [[ "$LAUNCHER" == "mpi" ]]; then
  [[ -n "$HOSTFILE" && -f "$HOSTFILE" ]] || { echo "LAUNCHER=mpi requires HOSTFILE; got: '$HOSTFILE'" >&2; exit 2; }
  command -v "$MPI_BIN" >/dev/null || { echo "MPI launcher '$MPI_BIN' not on PATH" >&2; exit 2; }
  NODES=$(grep -cE '^[^#[:space:]]' "$HOSTFILE")
  TOTAL_PROCS=$((NODES * GPUS_PER_NODE))
  echo "▣ LAUNCHER=mpi  hostfile=$HOSTFILE  nodes=$NODES  gpus/node=$GPUS_PER_NODE  total=$TOTAL_PROCS"
else
  echo "▣ LAUNCHER=local  (single-host synthetic; HCA loopback, not real fabric)"
fi

# ============================================================ Env ======
# Vars that are always set. Multi-node specifics added below.
NCCL_ENV=(
  LD_LIBRARY_PATH="$NCCL_ROOT/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
  NCCL_DEBUG=WARN
)

if [[ "$LAUNCHER" == "local" ]]; then
  # Synthetic single-host: force NET transport so the FC path runs.
  NCCL_ENV+=( NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 )
fi

case "$NIC_MODE" in
  spread)
    if [[ "$LAUNCHER" == "local" ]]; then
      # Curate 8 NICs for 8 ranks (1:1, no real incast on a single host)
      NCCL_ENV+=( NCCL_IB_HCA="^mlx5_8,mlx5_9,mlx5_10,mlx5_11" )
    fi
    # mpi mode: leave NIC selection alone — let NCCL use all configured HCAs
    ;;
  pin1)
    # All ranks share PIN_NIC. In mpi mode this creates true cross-node
    # incast (every cross-node flow hits this single NIC on each receiver).
    NCCL_ENV+=( NCCL_IB_HCA="=$PIN_NIC" )
    echo "▣ NIC_MODE=pin1 → NCCL_IB_HCA==$PIN_NIC (8:1 share, real incast in mpi mode)"
    if [[ "$FC_PIN_BW_GBS" -gt 0 ]]; then
      echo "▣ FC_PIN_BW_GBS=$FC_PIN_BW_GBS — fc-* configs override BW to compensate for unseen share"
    fi
    ;;
  *) echo "unknown NIC_MODE=$NIC_MODE (expected: spread | pin1)" >&2; exit 2 ;;
esac

# Per-config env overrides (these become extra "K=V K=V" added to each run).
declare -A CONFIGS
CONFIGS[baseline]="NCCL_RECV_FC_BW_GBS=1000000"
if [[ "$NIC_MODE" == "pin1" && "$FC_PIN_BW_GBS" -gt 0 ]]; then
  CONFIGS[fc-default]="NCCL_RECV_FC_BW_GBS=$FC_PIN_BW_GBS"
  CONFIGS[fc-guard]="NCCL_RECV_FC_BW_GBS=$FC_PIN_BW_GBS NCCL_RECV_FC_BDP_BYTES=1048576 NCCL_RECV_FC_PROACTIVE_RELEASE=1"
else
  CONFIGS[fc-default]=""
  CONFIGS[fc-guard]="NCCL_RECV_FC_BDP_BYTES=1048576 NCCL_RECV_FC_PROACTIVE_RELEASE=1"
fi

# ============================================================ Run ======
ALLTOALL_BIN="$NCCL_TESTS_ROOT/build/alltoall_perf"
[[ -x "$ALLTOALL_BIN" ]] || { echo "Missing $ALLTOALL_BIN — build nccl-tests first." >&2; exit 1; }

# Extract just the KEY names (no values) from "K=V K=V ..." for mpirun -x.
env_keys_from_kvs() {
  local out=()
  for kv in $1; do out+=( "${kv%%=*}" ); done
  printf '%s\n' "${out[@]}"
}

run_one_local() {
  local name=$1 extra=$2 trial=$3
  local log="$OUTDIR/${name}-t${trial}.log"
  echo "▶ $name (trial $trial, local) → $log"
  env "${NCCL_ENV[@]}" $extra \
    "$ALLTOALL_BIN" -b "$SIZE_MIN" -e "$SIZE_MAX" -f 2 \
                    -g 1 -t "$NGPU" -n "$ITERS" -w "$WARMUP" \
    > "$log" 2>&1
}

run_one_mpi() {
  local name=$1 extra=$2 trial=$3
  local log="$OUTDIR/${name}-t${trial}.log"
  echo "▶ $name (trial $trial, mpi $NODES×$GPUS_PER_NODE) → $log"

  # Build -x exports: every KEY from NCCL_ENV + every KEY from extra +
  # PATH so remote bash can find binaries.
  local x_args=( -x PATH )
  local k
  for kv in "${NCCL_ENV[@]}"; do x_args+=( -x "${kv%%=*}" ); done
  for kv in $extra; do x_args+=( -x "${kv%%=*}" ); done

  # `env` sets the values locally; mpirun -x forwards them to remote procs.
  env "${NCCL_ENV[@]}" $extra \
    "$MPI_BIN" --hostfile "$HOSTFILE" -np "$TOTAL_PROCS" \
      --map-by "ppr:${GPUS_PER_NODE}:node" --bind-to none \
      $MPI_EXTRA_ARGS \
      "${x_args[@]}" \
      "$ALLTOALL_BIN" -b "$SIZE_MIN" -e "$SIZE_MAX" -f 2 \
                      -g 1 -n "$ITERS" -w "$WARMUP" \
    > "$log" 2>&1
}

selected=()
case "$MODE" in
  all)        selected=(baseline fc-default fc-guard) ;;
  baseline)   selected=(baseline) ;;
  fc-default) selected=(fc-default) ;;
  fc-guard)   selected=(fc-guard) ;;
  *) echo "unknown MODE=$MODE" >&2; exit 2 ;;
esac

for cfg in "${selected[@]}"; do
  for t in $(seq 1 "$TRIALS"); do
    if [[ "$LAUNCHER" == "mpi" ]]; then
      run_one_mpi "$cfg" "${CONFIGS[$cfg]}" "$t"
    else
      run_one_local "$cfg" "${CONFIGS[$cfg]}" "$t"
    fi
  done
done

# ============================================================ Summary ==
python3 - "$OUTDIR" "$NIC_MODE" "$LAUNCHER" "$NODES" "${selected[@]}" <<'PY'
import os, re, sys, statistics, glob
outdir, nic_mode, launcher, nodes = sys.argv[1:5]
cfgs = sys.argv[5:]
ROW_RE = re.compile(r'^\s*(\d+)\s+\d+\s+\S+\s+\S+\s+\S*\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)')

data = {c: {} for c in cfgs}
for c in cfgs:
    for log in sorted(glob.glob(os.path.join(outdir, f"{c}-t*.log"))):
        with open(log) as f:
            for line in f:
                m = ROW_RE.match(line)
                if not m: continue
                size = int(m.group(1))
                busbw = float(m.group(4))
                data[c].setdefault(size, []).append(busbw)

sizes = sorted(set().union(*[d.keys() for d in data.values()]))
print()
print(f"LAUNCHER={launcher}  NODES={nodes}  NIC_MODE={nic_mode}  (busbw GB/s; median ± stddev across trials)")
print("─" * 116)
print(f"{'size':>10}  " + "  ".join(f"{c:>26}" for c in cfgs) + "    deltas vs baseline")
print("─" * 116)
for s in sizes:
    cells = []
    base = None
    for c in cfgs:
        vs = data[c].get(s, [])
        if not vs:
            cells.append(("—", None))
        else:
            med = statistics.median(vs)
            sd = statistics.stdev(vs) if len(vs) > 1 else 0.0
            cells.append((f"{med:5.2f}±{sd:4.2f}", med))
            if c == "baseline": base = med
    cell_strs = []
    deltas = []
    for c,(s_,v) in zip(cfgs, cells):
        cell_strs.append(f"{s_:>26}")
        if c != "baseline" and base and v:
            pct = (v - base) / base * 100
            deltas.append(f"{c}: {pct:+5.1f}%")
    sz = f"{s/1024:.0f}K" if s < 2**20 else (f"{s/2**20:.0f}M" if s < 2**30 else f"{s/2**30:.1f}G")
    print(f"{sz:>10}  " + "  ".join(cell_strs) + "    " + " | ".join(deltas))
print("─" * 116)
print()
print("notes:")
print(" • busbw = effective per-GPU bandwidth; higher is better. stddev across trials matters most under contention.")
print(" • baseline ≈ FC throttling off (BW_GBS=1000000), control plane still active for fair overhead comparison.")
print(" • spread = no artificial NIC sharing; pin1 = all ranks pinned to PIN_NIC (real cross-node incast in mpi mode).")
print(" • local mode = single-host synthetic (HCA loopback). mpi mode = real fabric — this is the meaningful test.")
PY

# ============================================================ Runbook ==
# How to run multi-node:
#
#   PRECONDITIONS (do once per cluster):
#     1. libnccl.so.2.30.4 (this branch's build) installed on every node at
#        the SAME absolute path. Either:
#          - NFS-mount $NCCL_ROOT/build/lib on all nodes, OR
#          - rsync to each node:
#              for h in node1 node2 ...; do
#                rsync -a $NCCL_ROOT/build/lib/  ${h}:$NCCL_ROOT/build/lib/
#                rsync -a $NCCL_TESTS_ROOT/build ${h}:$NCCL_TESTS_ROOT/
#              done
#     2. Passwordless SSH between launcher node and all workers
#          ssh-copy-id node1; ssh node1 true   # verify
#     3. mpirun (Open MPI ≥ 4) on the launcher node. Verify: mpirun --version
#     4. Same CUDA + driver version on all nodes (libcudart.so.13 must exist)
#     5. RDMA fabric reachable between nodes (test with ib_send_bw or similar)
#
#   HOSTFILE format (one host per line):
#       node-001 slots=8
#       node-002 slots=8
#       node-003 slots=8
#       node-004 slots=8
#
#   RUNS (from any one launcher node):
#     # baseline spread (let NCCL pick NICs naturally):
#     HOSTFILE=~/hosts.txt NIC_MODE=spread bash scripts/bench-fc-alltoall.sh
#
#     # 8:1 incast on mlx5_0 across all nodes — this is the FC stress test:
#     HOSTFILE=~/hosts.txt NIC_MODE=pin1 PIN_NIC=mlx5_0 \
#         bash scripts/bench-fc-alltoall.sh
#
#     # 8:1 with FC's BW manually corrected to match real per-rank share
#     # (NIC=200 Gbps, GPUS_PER_NODE=8 ⇒ per-rank claim = 25/8 ≈ 3 GB/s):
#     HOSTFILE=~/hosts.txt NIC_MODE=pin1 PIN_NIC=mlx5_0 FC_PIN_BW_GBS=3 \
#         bash scripts/bench-fc-alltoall.sh
#
#   TROUBLESHOOTING:
#     • "library not found" on remote → libnccl.so or libcudart not at same
#       path on workers. Verify with `ssh node-002 ls $LD_LIBRARY_PATH`.
#     • "MCA pml: ucx required" / IB errors → add MPI_EXTRA_ARGS="-mca pml ob1
#       -mca btl tcp,self" to fall back to TCP for MPI bootstrap (NCCL still
#       uses IB for data); or "-mca pml ucx" if UCX is the cluster default.
#     • Bootstrap timeout → set NCCL_SOCKET_IFNAME locally in NCCL_ENV (above)
#       to the management NIC name (e.g. eth0), then add "-x NCCL_SOCKET_IFNAME"
#     • To confirm FC is wired up, run once with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET
#       and grep for "FlowControl rank … registered NIC dev=…"
#     • To compare FC-on vs FC-truly-off (not just override-disabled), build
#       master into a separate dir and switch LD_LIBRARY_PATH between runs.
