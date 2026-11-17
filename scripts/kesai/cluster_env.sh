# Our clusters on top of scripts/slurm/job_env.sh, sourced after the #SBATCH block by a job submitted from the checkout root:
#   source scripts/kesai/cluster_env.sh || exit 1
# Sets CLUSTER (kesai | scaleway | local), HOME_BASE (other users' homes: $HOME_BASE/<user>) and SHARED_DATA (nuplan,
# nurec, ...), then everything job_env.sh sets (PD, SLURM_CPUS_PER_TASK, numa_pin_for_gpu, pick_port_base).
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
export CLUSTER HOME_BASE SHARED_DATA
source scripts/slurm/job_env.sh || return 1 2>/dev/null || exit 1
echo "cluster: $CLUSTER  shared data: ${SHARED_DATA:-none}"
