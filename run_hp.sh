#!/usr/bin/env bash
#SBATCH --job-name=hp_sweep
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --account=--
# 13 envs * 10 seeds * 1 variants = 130 tasks
#SBATCH --array=0-129

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
    #"Isaac-Velocity-Rough-Unitree-Go2-v0"
    #"Throwing-G1-General"
    #"Isaac-Velocity-Rough-H1-v0"
)

NUM_ENVS=${#ENVS[@]}
NUM_SEEDS=10
NUM_VARIANTS=1  # 0 = baseline, 1 = critic_multi + pcgrad + normpres
PER_ENV=$((NUM_SEEDS * NUM_VARIANTS))
TOTAL_TASKS=$((NUM_ENVS * PER_ENV))

# Bounds check
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range [0,$((TOTAL_TASKS-1))]"
    exit 1
fi

# Indices
ENV_IDX=$(( SLURM_ARRAY_TASK_ID / PER_ENV ))
REM=$(( SLURM_ARRAY_TASK_ID % PER_ENV ))
SEED_IDX=$(( REM / NUM_VARIANTS ))
VARIANT_IDX=$(( REM % NUM_VARIANTS ))

ENV_NAME=${ENVS[$ENV_IDX]}

# Variant flags
if [[ $VARIANT_IDX -eq 1 ]]; then
    EXTRA_ARGS=""
    VARIANT_TAG="baseline"
else
    EXTRA_ARGS="--use_critic_multi --use_pcgrad_full" 
    VARIANT_TAG="pcgrad_combo"
fi

# Deterministic base seed per env, consistent across variants; offset by SEED_IDX to get 5 seeds
JOB_ANCHOR="${SLURM_ARRAY_JOB_ID:-LOCAL}"
HASH8=$(printf "%s" "${JOB_ANCHOR}_${ENV_NAME}" | sha256sum | cut -d' ' -f1 | head -c 8)
BASE_SEED=$(( 16#${HASH8} % 1001 ))
SEED=$(( (BASE_SEED + SEED_IDX) % 1001 ))

echo "ENV_IDX=$ENV_IDX SEED_IDX=$SEED_IDX VARIANT_IDX=$VARIANT_IDX -> ENV=$ENV_NAME SEED=$SEED VARIANT=$VARIANT_TAG"
echo "Running task $SLURM_ARRAY_TASK_ID: ENV=$ENV_NAME, SEED=$SEED, VARIANT=$VARIANT_TAG"

# Run the experiment (no entropy argument)
./isaaclab.sh -p ./scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$ENV_NAME" \
    --headless \
    --num_envs 4096 \
    --seed "$SEED" \
    $EXTRA_ARGS
