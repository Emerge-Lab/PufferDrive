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
    # multi-node NCCL: RDMA (RoCE v2) on the 400G GPU network since 18 Nov 2026. mlx5_0 is the GPU-network card of the
    # RTX nodes, GID index 3 its RoCE v2 entry with the IPv4 address (the cluster's login default, repeated here for
    # jobs that do not start from a login shell). NCCL_SOCKET_IFNAME above stays: NCCL uses it for its bootstrap.
    # NCCL_IB_DISABLE=1 in the environment of a job forces sockets.
    export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0} NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
    # direct GPU-to-GPU copies between all GPUs of a node (cluster default since 18 Nov 2026: 8-GPU all_reduce
    # 13.3 -> 21-22 GB/s), and NCCL's tree pattern for jobs on more than one node (256 MiB all_reduce with direct
    # copies: two nodes 11.4 -> 18.5 GB/s, six nodes 10.3 -> 13.0; inside one node the tree is slower, so only there)
    export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}
    [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ] && export NCCL_ALGO=${NCCL_ALGO:-Tree}
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
