#!/bin/bash
#SBATCH --job-name eval_nuplan_nr
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 1-00:00
#SBATCH --gres gpu:8
#SBATCH --mem=0
#SBATCH --exclusive                      # whole node on either cluster, --mem=0 = all of its memory
#SBATCH --output experiments/logs/nuPlan_nonreactive_%j.out
#SBATCH --error experiments/logs/nuPlan_nonreactive_%j.err
#SBATCH --partition dev
# nuPlan Val14 closed-loop evaluation against non-reactive log-replay agents (CLS-NR): 9_nuPlan.sh with the non-reactive
# challenge, so the reactive and non-reactive scores always come from the same planner settings. Overridable env: as in
# 9_nuPlan.sh; results go to $RUN_DIR/eval/nuplan_val14_nonreactive_*.
source scripts/kesai/cluster_env.sh || exit 1
CHALLENGES=closed_loop_nonreactive_agents_pufferdrive exec bash "$PD/scripts/kesai/9_nuPlan.sh"
