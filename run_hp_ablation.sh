#!/usr/bin/env bash
#SBATCH --job-name=hp_sweep
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --account=---
# 13 envs * 3 seeds * 1 variant * 10 entropies = 390 tasks
#SBATCH --array=0-389

# Environments
ENVS=(
    #"Isaac-Repose-Cube-Shadow-Direct-v0"
    #"Isaac-Humanoid-v0"
    #"Isaac-Velocity-Rough-G1-v0"
    #"Isaac-Ant-v0"
    #"Isaac-Tracking-LocoManip-Digit-v0"
    #"Isaac-Repose-Cube-Allegro-v0"
    #"Isaac-Open-Drawer-Franka-v0"
    #"Isaac-Lift-Cube-Franka-v0"
    #"Isaac-Reach-UR10-v0"
    #"Isaac-Reach-Franka-v0"
    #"Isaac-Quadcopter-Direct-v0"
    #"Isaac-Velocity-Rough-Anymal-D-v0"
    #"Isaac-Velocity-Rough-Unitree-Go2-v0"
    #"Throwing-G1-General"
    #"Isaac-Velocity-Rough-H1-v0"
)

# Entropies
ENTROPIES=(
"0.00010000"
"0.00018847"
"0.00035520"
"0.00066943"
"0.00126166"
"0.00237782"
"0.00448140"
"0.00844598"
"0.01591789"
"0.03000000"
)

NUM_ENVS=${#ENVS[@]}
NUM_SEEDS=3
NUM_VARIANTS=1  # only pcgrad variant
NUM_ENTROPIES=${#ENTROPIES[@]}
PER_ENV=$((NUM_SEEDS * NUM_VARIANTS * NUM_ENTROPIES))
TOTAL_TASKS=$((NUM_ENVS * PER_ENV))

# Bounds check
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
        echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range [0,$((TOTAL_TASKS-1))]"
        exit 1
fi

# Indices
ENV_IDX=$(( SLURM_ARRAY_TASK_ID / PER_ENV ))
REM=$(( SLURM_ARRAY_TASK_ID % PER_ENV ))

SEED_BLOCK=$(( NUM_VARIANTS * NUM_ENTROPIES ))
SEED_IDX=$(( REM / SEED_BLOCK ))
REM2=$(( REM % SEED_BLOCK ))

VARIANT_BLOCK=$(( NUM_ENTROPIES ))
VARIANT_IDX=$(( REM2 / VARIANT_BLOCK ))
ENTROPY_IDX=$(( REM2 % VARIANT_BLOCK ))

ENV_NAME=${ENVS[$ENV_IDX]}
ENTROPY="${ENTROPIES[$ENTROPY_IDX]}"

# Variant flags: only pcgrad
if [[ $VARIANT_IDX -eq 0 ]]; then
        EXTRA_ARGS="--use_critic_multi --use_pcgrad"
        VARIANT_TAG="pcgrad_combo"
else
        EXTRA_ARGS=""
        VARIANT_TAG="baseline"
fi

# Deterministic base seed per env, consistent across variants; offset by SEED_IDX
JOB_ANCHOR="${SLURM_ARRAY_JOB_ID:-LOCAL}"
HASH8=$(printf "%s" "${JOB_ANCHOR}_${ENV_NAME}" | sha256sum | cut -d' ' -f1 | head -c 8)
BASE_SEED=$(( 16#${HASH8} % 1001 ))
SEED=$(( (BASE_SEED + SEED_IDX) % 1001 ))

echo "ENV_IDX=$ENV_IDX SEED_IDX=$SEED_IDX VARIANT_IDX=$VARIANT_IDX ENTROPY_IDX=$ENTROPY_IDX -> ENV=$ENV_NAME SEED=$SEED VARIANT=$VARIANT_TAG ENTROPY=$ENTROPY"
echo "Running task $SLURM_ARRAY_TASK_ID: ENV=$ENV_NAME, SEED=$SEED, VARIANT=$VARIANT_TAG, ENTROPY=$ENTROPY"

# Run the experiment
./isaaclab.sh -p ./scripts/reinforcement_learning/rsl_rl/train.py \
        --task "$ENV_NAME" \
        --headless \
        --num_envs 4096 \
        --seed "$SEED" \
        --entropy_coef "$ENTROPY" \
        $EXTRA_ARGS
