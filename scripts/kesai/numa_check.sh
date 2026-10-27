#!/bin/bash
# Where a running training job's env workers execute relative to their rank's GPU socket on a kesai node
# (GPUs 0-3 on NUMA node 0, 4-7 on node 1). Run on the compute node: bash scripts/kesai/numa_check.sh
pids=$(pgrep -f "pufferlib.pufferl train")
for pid in $pids; do
    ppid=$(ps -o ppid= -p "$pid" | tr -d ' ')
    echo "$pids" | grep -qx "$ppid" && continue   # a worker: its parent is a trainer process
    rank=$(tr '\0' '\n' < /proc/"$pid"/environ 2>/dev/null | awk -F= '$1 == "LOCAL_RANK" {print $2}')
    [ -n "$rank" ] || continue
    node0=0; node1=0
    for kid in $(pgrep -P "$pid"); do
        psr=$(ps -o psr= -p "$kid" | tr -d ' ')
        [ -n "$psr" ] || continue
        if [ "$psr" -lt 128 ] || { [ "$psr" -ge 256 ] && [ "$psr" -lt 384 ]; }; then node0=$((node0 + 1)); else node1=$((node1 + 1)); fi
    done
    echo "rank $rank (GPU on node $((rank / 4))): $node0 workers on node 0, $node1 on node 1; trainer mask: $(taskset -cp "$pid" | cut -d: -f2)"
done
