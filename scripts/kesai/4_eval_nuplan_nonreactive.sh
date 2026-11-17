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
# nuPlan Val14 eval against non-reactive log-replay agents (CLS-NR): 3_eval_nuplan_reactive.sh (same env knobs and planner settings)
# with the non-reactive challenge; results go to $RUN_DIR/eval/nuplan_val14_nonreactive_*.
source scripts/kesai/cluster_env.sh || exit 1
CHALLENGES=closed_loop_nonreactive_agents_opengigaflow exec bash "$PD/scripts/kesai/3_eval_nuplan_reactive.sh"
