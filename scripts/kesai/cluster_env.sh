# Cluster detection for the scripts in this folder, sourced from the checkout root after the #SBATCH block:
#   source scripts/kesai/cluster_env.sh || exit 1
# Always submit from the checkout root (sbatch keeps it as the job's working directory) and create experiments/logs once.
# Sets CLUSTER (kesai | scaleway | local), HOME_BASE (where user homes live; other users' paths are $HOME_BASE/<user>/...),
# SHARED_DATA (the shared datasets: nuplan, nurec, ... live directly below it), PD (this checkout) and SLURM_CPUS_PER_TASK
# (whole-node jobs carry #SBATCH --exclusive instead of a per-cluster CPU count; srun steps and scripts/parallel_eval.py
# read the count from here). Your own paths stay $HOME/..., which is /home/<user> on scaleway and /cephfs/home/<user> on kesai.
if [ -d /cephfs/data ]; then
    CLUSTER=kesai
    HOME_BASE=/cephfs/home
    SHARED_DATA=/cephfs/data
    # the nodes also carry the BMC's USB NIC (169.254.x.x): pin NCCL and Gloo to the cluster network
    CLUSTER_IFACE=$(ip -o -4 addr show to 192.168.20.0/24 2>/dev/null | head -n 1 | cut -d' ' -f2)
    [ -n "$CLUSTER_IFACE" ] && export NCCL_SOCKET_IFNAME=$CLUSTER_IFACE GLOO_SOCKET_IFNAME=$CLUSTER_IFACE
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
    echo "cluster_env.sh: $PD is not the PufferDrive checkout; submit from the checkout root or set PD" >&2
    return 1 2>/dev/null || exit 1
fi
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-$(nproc)}}
export CLUSTER HOME_BASE SHARED_DATA PD
echo "cluster: $CLUSTER  checkout: $PD  shared data: ${SHARED_DATA:-none}  cpus per task: $SLURM_CPUS_PER_TASK"
