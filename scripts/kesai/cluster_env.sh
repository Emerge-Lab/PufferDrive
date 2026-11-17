# Cluster detection for jobs submitted from the checkout root (create experiments/logs once), after the #SBATCH block:
#   source scripts/kesai/cluster_env.sh || exit 1
# Sets CLUSTER (kesai | scaleway | local), HOME_BASE (other users' homes: $HOME_BASE/<user>), SHARED_DATA (nuplan,
# nurec, ...), PD (this checkout) and SLURM_CPUS_PER_TASK (the node's CPUs: whole-node jobs carry --exclusive).
if [ -d /cephfs/data ]; then
    CLUSTER=kesai
    HOME_BASE=/cephfs/home
    SHARED_DATA=/cephfs/data
    # the nodes also carry the BMC's USB NIC (169.254.x.x): pin NCCL and Gloo to the cluster network
    CLUSTER_IFACE=$(ip -o -4 addr show to 192.168.20.0/24 2>/dev/null | head -n 1 | cut -d' ' -f2)
    [ -n "$CLUSTER_IFACE" ] && export NCCL_SOCKET_IFNAME=$CLUSTER_IFACE GLOO_SOCKET_IFNAME=$CLUSTER_IFACE
    # the 400G NICs have no working fabric yet: NCCL must not try RoCE over them (remove with the fabric)
    export NCCL_IB_DISABLE=1
elif [ -d /home/shared/data ]; then
    CLUSTER=scaleway
    HOME_BASE=/home
    SHARED_DATA=/home/shared/data
else
    CLUSTER=local
    HOME_BASE=/home
    SHARED_DATA=
fi
PD=${PD:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -f "$PD/setup.py" ]; then
    echo "cluster_env.sh: $PD is not the OpenGigaFlow checkout; submit from the checkout root or set PD" >&2
    return 1 2>/dev/null || exit 1
fi
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-$(nproc)}}
# opengigaflow opens 4 pipe fds per env worker; the job's soft nofile limit (1024 on kesai) caps workers near 200 per rank
ulimit -n "$(ulimit -Hn)" 2>/dev/null || echo "cluster_env.sh: could not raise the open-file limit (soft $(ulimit -Sn), hard $(ulimit -Hn))" >&2
export CLUSTER HOME_BASE SHARED_DATA PD
echo "cluster: $CLUSTER  checkout: $PD  shared data: ${SHARED_DATA:-none}  cpus per task: $SLURM_CPUS_PER_TASK"

# "taskset -c <cpus>" for the NUMA node of GPU index $1 (nvidia-smi order), empty when unknown; prefix per-GPU workers with it
numa_pin_for_gpu() {
    local bus node
    bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$1" 2>/dev/null | tr 'A-F' 'a-f')
    node=$(cat "/sys/bus/pci/devices/${bus:4}/numa_node" 2>/dev/null) || return 0
    [ "$node" -ge 0 ] 2>/dev/null && echo "taskset -c $(cat "/sys/devices/system/node/node${node}/cpulist")"
}

# first of the port bases $2... whose rpc (base+50w), streaming (rpc+1) and TM (rpc+6000) ports of $1 workers are all free
pick_port_base() {
    local worker_count=$1 listening base w port busy
    shift
    listening=" $(ss -ltun 2>/dev/null | awk 'NR > 1 {sub(/.*:/, "", $5); print $5}' | sort -u | tr '\n' ' ') "
    for base in "$@"; do
        busy=0
        for ((w = 0; w < worker_count; w++)); do
            for port in $((base + 50 * w)) $((base + 50 * w + 1)) $((base + 6000 + 50 * w)); do
                [[ "$listening" == *" $port "* ]] && { busy=$port; break 2; }
            done
        done
        [ "$busy" = 0 ] && { echo "$base"; return 0; }
        echo "port base $base: port $busy is in use, trying the next range" >&2
    done
    return 1
}
