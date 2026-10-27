#!/bin/bash
# Where a running training job's env workers execute relative to their rank's GPU socket on a kesai node
# (GPUs 0-3 on NUMA node 0, 4-7 on node 1). Run on the compute node: bash scripts/kesai/numa_check.sh
# A trainer rank is a process with LOCAL_RANK in its environment that has children (the env workers);
# torchrun has children but no LOCAL_RANK, workers have LOCAL_RANK but no children.
found=0
# torchrun's children are the trainer ranks; without torchrun fall back to scanning every pufferl process (slow: ~500 workers)
candidates=$(for launcher in $(pgrep -f "bin/torchrun"); do pgrep -P "$launcher"; done)
[ -n "$candidates" ] || candidates=$(pgrep -f "pufferlib.pufferl")
for pid in $candidates; do
    rank=$(tr '\0' '\n' < /proc/"$pid"/environ 2>/dev/null | awk -F= '$1 == "LOCAL_RANK" {print $2}')
    [ -n "$rank" ] || continue
    kids=$(pgrep -P "$pid")
    [ -n "$kids" ] || continue
    found=1
    node0=0; node1=0
    for kid in $kids; do
        psr=$(ps -o psr= -p "$kid" | tr -d ' ')
        [ -n "$psr" ] || continue
        if [ "$psr" -lt 128 ] || { [ "$psr" -ge 256 ] && [ "$psr" -lt 384 ]; }; then node0=$((node0 + 1)); else node1=$((node1 + 1)); fi
    done
    echo "rank $rank (GPU on node $((rank / 4))): $node0 workers on node 0, $node1 on node 1; trainer mask:$(taskset -cp "$pid" | cut -d: -f2)"
done
[ "$found" = 1 ] || echo "no pufferlib.pufferl trainer processes with LOCAL_RANK found on $(hostname); is the training job on this node?"
