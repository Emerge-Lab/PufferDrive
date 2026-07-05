cd /scratch/yw4142/PufferDrive4

uv venv /scratch/yw4142/PufferDrive4/venv
source /scratch/yw4142/PufferDrive4/venv/bin/activate
export UV_CACHE_DIR=/scratch/yw4142/.cache/uv
uv pip install -e .

# Setup
./scripts/setup_container.sh create-overlay
sbatch --account=torch_pr_37_lpinto --gres=gpu:1 --cpus-per-task=8 --mem=32gb --time=60 \
    --wrap "./scripts/setup_container.sh install"

# Per code build
sbatch --account=torch_pr_355_tandon_advanced --gres=gpu:1 --cpus-per-task=4 --mem=8gb --time=15 \
    --wrap "./scripts/setup_container.sh rebuild"

# Submit job
python scripts/submit_cluster.py \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --program_config scripts/cluster_configs/train_base.yaml \
    --save_dir /scratch/yw4142/PufferDrive_exp/random_goal --prefix $(date +%Y-%m-%d_%H-%M-%S)_multi_agent \
    --container --time 1440 \
    --args train.seed=42 env.num_target_waypoints=1 env.goal_on_lane=False env.min_waypoint_spacing=6.0 env.max_waypoint_spacing=500.0 env.obs_norm_goal_offset_m=500.0
python scripts/submit_cluster.py \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --program_config scripts/cluster_configs/train_base.yaml \
    --save_dir /scratch/yw4142/PufferDrive_exp/random_goal --prefix $(date +%Y-%m-%d_%H-%M-%S)_multi_agent \
    --container --time 1440 \
    --args train.seed=42 env.num_target_waypoints=3 env.goal_on_lane=False env.min_waypoint_spacing=6.0 env.max_waypoint_spacing=500.0 env.obs_norm_goal_offset_m=500.0
python scripts/submit_cluster.py \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --program_config scripts/cluster_configs/train_base.yaml \
    --save_dir /scratch/yw4142/PufferDrive_exp/random_goal --prefix $(date +%Y-%m-%d_%H-%M-%S)_multi_agent \
    --container --time 1440 \
    --args train.seed=42 

     --args env.enable-lane-change-goals=True env.obs-dropout-lane=0.4 env.obs-dropout-boundary=0.4 env.partner-blindness-prob=0.05 env.phantom-braking-prob=0.1 env.phantom-braking-trigger-prob=0.1 \


<!-- python scripts/submit_cluster.py \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --program_config scripts/cluster_configs/train_base.yaml \
    --save_dir experiments/mix_train \
    --container -->
    
    --args load_model_path=/path/to/checkpoint.pt

# Modular job
chmod +x ./yvonne/singularity_run.sh
./yvonne/singularity_run.sh

## Eval
python -m pufferlib.pufferl eval puffer_drive \
    --load-model-path /scratch/yw4142/PufferDrive_exp/random_goal/2026-06-24_00-34-36_multi_agent_train_base_seed0_e751d4d/puffer_drive_5lezrvzr/models/model_puffer_drive_000200.pt \
    --evaluator dnf_triage

## Train
python -m pufferlib.pufferl train puffer_drive \
    --eval.validation-defaults.interval=4 \
    --env.obs-dropout-lane=0.4 \
    --env.obs-dropout-boundary=0.4 \
    --env.partner-blindness-prob=0.05 \
    --env.phantom-braking-prob=0.1 \
    --env.phantom-braking-trigger-prob=0.1

### Train single agent
python scripts/submit_cluster.py \
    --save_dir /scratch/yw4142/PufferDrive_exp/random_goal --prefix $(date +%Y-%m-%d_%H-%M-%S)_single_agent \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --program_config scripts/cluster_configs/single_agent_speed_run.yaml \
    --container --time 720 \
    --args train.seed=0 env.num_target_waypoints=1 train.total_timesteps=20000000000
python scripts/submit_cluster.py \
    --save_dir /scratch/yw4142/PufferDrive_exp/random_goal --prefix $(date +%Y-%m-%d_%H-%M-%S)_single_agent \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --program_config scripts/cluster_configs/single_agent_speed_run.yaml \
    --container --time 720 \
    --args train.seed=0 env.num_target_waypoints=3 train.total_timesteps=20000000000
    
validation_gigaflow
validation_replay
    behaviors_full_dir
    behaviors_hard_stop
    behaviors_highway_straight
    behaviors_lane_change
    behaviors_merge
    behaviors_parked_cars
    behaviors_roundabout
    behaviors_stopped_traffic
    behaviors_traffic_light_green
    behaviors_traffic_light_stop
    behaviors_unprotected_left
    behaviors_unprotected_right
    behaviors_wosac
    gigaflow_full_dir
    gigaflow_wosac

MODEL=/scratch/yw4142/PufferDrive4/weights/tomate/models/model_puffer_drive_005000.pt
evaluators=(
    validation_gigaflow
)
for evaluator in $evaluators; do
    python -m pufferlib.pufferl eval puffer_drive \
        --load-model-path "$MODEL" \
        --evaluator "$evaluator" \
        > "/scratch/yw4142/PufferDrive4/benchmark/tomate_${evaluator}.log" 2>&1
done


# control_vehicles
validation_gigaflow
# control_sdc_only
<!-- behaviors_defaults -->
behaviors_full_dir
behaviors_hard_stop
behaviors_highway_straight
behaviors_lane_change
behaviors_merge
behaviors_parked_cars
behaviors_roundabout
behaviors_stopped_traffic
behaviors_traffic_light_green
behaviors_traffic_light_stop
behaviors_unprotected_left
behaviors_unprotected_right
# control_vehicles
wosac <!-- behaviors_wosac -->

## Train
python -m pufferlib.pufferl train puffer_drive --wandb --wandb-project pufferdrive 

python -m pufferlib.pufferl train puffer_drive --wandb --wandb-project pufferdrive --train.data-dir experiments/train_nuplan --env.map-dir /scratch/ev2237/data/nuplan/nuplan_mini_train_bins --env.simulation-mode replay --env.control-mode control_vehicles --env.num-maps 876 --env.num-agents 256 --load-model-path /scratch/yw4142/PufferDrive4/experiments/train_nuplan/puffer_drive_wx8btcui/models/model_puffer_drive_000550.pt

python -m pufferlib.pufferl train puffer_drive --wandb --wandb-project pufferdrive --train.data-dir experiments/fintune_nuplan --env.map-dir /scratch/ev2237/data/nuplan/nuplan_mini_train_bins --env.simulation-mode replay --env.control-mode control_vehicles --env.num-maps 876 --env.num-agents 256 --load-model-path /scratch/yw4142/PufferDrive4/experiments/fintune_nuplan/puffer_drive_z5tffzt1/models/model_puffer_drive_000250.pt --env.lane-segment-dropout 0.4 --env.boundary-segment-dropout 0.4 --env.partner-blindness-prob 0.05 --env.phantom-braking-prob 0.1 --env.phantom-braking-trigger-prob 0.1

## 2 mode
### fix map
env.simulation_mode = "replay"
env.control_mode = "control_sdc_only"
env.init_mode = "create_all_valid"
env.map_dir = "/scratch/ev2237/data/womd/training"
env.num_maps = 6724

### infinite map
env.simulation_mode = "gigaflow"
env.control_mode = "control_vehicles"
env.init_mode = "create_all_valid"
env.map_dir = "/scratch/ev2237/data/py123d_workdir/maps/opendrive"
env.num_maps = 8

# Data
/scratch/ev2237/data/
├── carla/                                  # 8 .bin files
│   ├── opendrive__Town01.bin
│   ├── opendrive__Town02.bin
│   ├── opendrive__Town03.bin
│   ├── opendrive__Town04.bin
│   ├── opendrive__Town05.bin
│   ├── opendrive__Town06.bin
│   ├── opendrive__Town07.bin
│   └── opendrive__Town10HD.bin
│
├── nuplan/
│   ├── categories/                         # 12 scenario types, ~32 files total
│   │   ├── hard_stop/                      #  1 file
│   │   ├── highway_straight/               # 10 files
│   │   ├── lane_change/                    #  4 files
│   │   ├── merge/                          #  1 file
│   │   ├── parked_cars/                    #  1 file
│   │   ├── right_turn/                     #  0 files
│   │   ├── roundabout/                     #  3 files
│   │   ├── stopped_traffic/                #  1 file
│   │   ├── traffic_light_green/            #  3 files
│   │   ├── traffic_light_stop/             #  5 files
│   │   ├── unprotected_left/               #  2 files
│   │   └── unprotected_right/              #  1 file
│   ├── categories_v021/                    # 11 scenario types, ~25 files total
│   │   ├── hard_stop/                      #  1 file
│   │   ├── highway_straight/               #  8 files
│   │   ├── lane_change/                    #  3 files
│   │   ├── merge/                          #  1 file
│   │   ├── parked_cars/                    #  1 file
│   │   ├── roundabout/                     #  1 file
│   │   ├── stopped_traffic/                #  1 file
│   │   ├── traffic_light_green/            #  2 files
│   │   ├── traffic_light_stop/             #  4 files
│   │   ├── unprotected_left/               #  2 files
│   │   └── unprotected_right/              #  1 file
│   ├── data/cache/                         # 64 files
│   ├── maps/                               # empty
│   ├── nuplan-maps-v1.0/                   # 4 cities + index JSON (7 files total)
│   │   ├── sg-one-north/9.17.1964/
│   │   ├── us-ma-boston/9.12.1817/
│   │   ├── us-nv-las-vegas-strip/9.15.1915/
│   │   └── us-pa-pittsburgh-hazelwood/9.17.1937/
│   ├── nuplan-v1.1/splits/                 # empty
│   ├── nuplan_full_dir_50/                 # 50 nuplan__*.bin files
│   ├── nuplan_full_dir_100/                # 100 nuplan__*.bin files
│   ├── nuplan_mini_train/                  # mirror of nuplan layout (data/, maps/, nuplan-maps-v1.0/, nuplan-v1.1/)
│   ├── nuplan_mini_train_bins/             # 877 files (nuplan__*.bin + failures.jsonl)
│   ├── nuplan_v021_100_subset/             # 100 nuplan__*.bin files
│   └── py123d_output/                      # 261 files total
│       ├── logs/                           #   3 files
│       └── maps/nuplan/                    #   1 directory
│
├── py123d_workdir/
│   └── maps/opendrive/                     # 8 .arrow files (Town01-07, Town10HD)
│
├── torc/
│   ├── tu24q4_bins/
│   │   ├── binaries_train/                 # 39,661 files
│   │   ├── binaries_test/                  #  4,957 files
│   │   └── binaries_val/                   #  4,957 files
│   └── tu24q4_bins_route_speed_Apr_17/     # 49,575 files
│
├── womd/
│   └── training/                           # 6,724 files (tfrecord-{00000..00999}-of-01000_*.bin)
│
└── xodr/                                   # 8 .xodr files (Town01-07, Town10HD)

# Notebook
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install ipykernel jupyter

python -m ipykernel install --user \
  --name pufferdrive4 \
  --display-name "Python3.12 (PufferDrive4)"