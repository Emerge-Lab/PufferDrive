# Environment of the scripts/slurm jobs, sourced after the #SBATCH block by a job submitted from the checkout root:
#   source scripts/slurm/job_env.sh || exit 1
# Sets PD (this checkout) and SLURM_CPUS_PER_TASK (the node's CPUs: the jobs take whole nodes), raises the open-file limit
# and defines numa_pin_for_gpu and pick_port_base.
PD=${PD:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -f "$PD/setup.py" ]; then
    echo "job_env.sh: $PD is not the OpenGigaFlow checkout; submit from the checkout root or set PD" >&2
    return 1 2>/dev/null || exit 1
fi
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-$(nproc)}}
# opengigaflow opens 4 pipe fds per env worker; a soft nofile limit of 1024 caps workers near 200 per rank
ulimit -n "$(ulimit -Hn)" 2>/dev/null || echo "job_env.sh: could not raise the open-file limit (soft $(ulimit -Sn), hard $(ulimit -Hn))" >&2
export PD
# once per job: a site script and the job it starts both source this file
if [ "${JOB_ENV_PRINTED_FOR:-}" != "${SLURM_JOB_ID:-local}" ]; then
    echo "checkout: $PD  cpus per task: $SLURM_CPUS_PER_TASK"
    export JOB_ENV_PRINTED_FOR=${SLURM_JOB_ID:-local}
fi

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
