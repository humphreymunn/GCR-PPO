#!/usr/bin/env bash
#SBATCH --job-name=hp_sweep
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --account=---
# 1 env * 1 seed-dim * 1 variants * 6 cardinalities * 20 combos = 200 tasks
#SBATCH --array=120-199

# -------- Config --------
ENVS=(
  "Throwing-G1-General"
  "Isaac-Humanoid-Direct-v0"  
)

NUM_ENVS=${#ENVS[@]}
NUM_SEEDS=1
NUM_VARIANTS=1
NUM_CARDINALITIES=6
COMBOS_PER_CARD=20

PER_ENV=$((NUM_SEEDS * NUM_VARIANTS * NUM_CARDINALITIES * COMBOS_PER_CARD))
TOTAL_TASKS=$((NUM_ENVS * PER_ENV))

# Bounds check
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range [0,$((TOTAL_TASKS-1))]"
  exit 1
fi

# -------- Indexing --------
ENV_IDX=$(( SLURM_ARRAY_TASK_ID / PER_ENV ))
REM=$(( SLURM_ARRAY_TASK_ID % PER_ENV ))

SEED_IDX=$(( REM / (NUM_VARIANTS * NUM_CARDINALITIES * COMBOS_PER_CARD) ))
REM=$(( REM % (NUM_VARIANTS * NUM_CARDINALITIES * COMBOS_PER_CARD) ))

VARIANT_IDX=$(( REM / (NUM_CARDINALITIES * COMBOS_PER_CARD) ))
REM=$(( REM % (NUM_CARDINALITIES * COMBOS_PER_CARD) ))

CARD_IDX=$(( REM / COMBOS_PER_CARD ))         # 0..3
COMBO_IDX=$(( REM % COMBOS_PER_CARD ))        # 0..19

ENV_NAME=${ENVS[$ENV_IDX]}
CARDINALITY=$(( CARD_IDX + 1 ))               # 1..4
JOB_ANCHOR="${SLURM_ARRAY_JOB_ID:-LOCAL}"

# Variant flags
if [[ $VARIANT_IDX -eq 1 ]]; then
  EXTRA_ARGS=""
  VARIANT_TAG="baseline"
else
  EXTRA_ARGS="--use_critic_multi"
  VARIANT_TAG="pcgrad_combo"
fi

# -------- Seeds (export before heredocs!) --------
export ENV_NAME VARIANT_IDX CARD_IDX COMBO_IDX JOB_ANCHOR

# Training seed: depends on variant so each method uses a different seed
SEED=$(
python3 - <<'PY'
import os, hashlib
s = f"{os.environ.get('JOB_ANCHOR','LOCAL')}|" \
    f"{os.environ.get('ENV_NAME','ENV')}|" \
    f"var={os.environ.get('VARIANT_IDX','0')}|" \
    f"card={os.environ.get('CARD_IDX','0')}|" \
    f"combo={os.environ.get('COMBO_IDX','0')}"
h = hashlib.sha256(s.encode()).digest()
seed = int.from_bytes(h[:8], 'big') % (2**32 - 1)   # in [0, 2**32-2]
print(seed)
PY
)

# Reward subset & per-component indices: same across variants
REWARD_ARGS=$(
python3 - <<'PY'
import os, hashlib, random
env      = os.environ.get('ENV_NAME', 'ENV')
job_anc  = os.environ.get('JOB_ANCHOR', 'LOCAL')
card_idx = int(os.environ.get('CARD_IDX', '0'))
combo_ix = int(os.environ.get('COMBO_IDX', '0'))

# Seed independent of variant so combos match across methods
h = hashlib.sha256(f"{job_anc}|{env}|card={card_idx}|combo={combo_ix}".encode()).digest()
rng = random.Random(int.from_bytes(h[:8], 'big'))

if env == "Throwing-G1-General":
    components = ['baseh_rew','hiporr_rew','armswlft_rew','armswrgt_rew','stephlft_rew','stephrgt_rew']
elif env == "Isaac-Humanoid-Direct-v0":
    components = ['baseh_rew','gait_rew','energy_rew','armsp_rew']
    
k = card_idx + 1
subset = rng.sample(components, k)
vals = {c: -1 for c in components}
for c in subset:
    vals[c] = rng.randint(0, 4)   # index 0..4

print(' '.join(f"--{c} {vals[c]}" for c in components))
PY
)

echo "TASK $SLURM_ARRAY_TASK_ID:"
echo "  ENV=$ENV_NAME  VARIANT=$VARIANT_TAG  SEED=$SEED"
echo "  cardinality=$CARDINALITY  combo_idx=$COMBO_IDX"
echo "  reward_args: $REWARD_ARGS"

# -------- Run --------
./isaaclab.sh -p ./scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$ENV_NAME" \
  --headless \
  --num_envs 4096 \
  --seed "$SEED" \
  $EXTRA_ARGS \
  $REWARD_ARGS
