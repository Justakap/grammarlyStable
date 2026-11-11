#!/bin/bash
set -e
cd "$(dirname "$0")"

RUNS=${RUNS:-"D C B"}
SEQUENTIAL=${SEQUENTIAL:-0}
BASELINES=${BASELINES:-1}
export A2_ADAPTER=${A2_ADAPTER:-qwen_4b_hinglish_lora}

if [ ! -f "$A2_ADAPTER/adapter_config.json" ]; then
  echo "Assignment 2 adapter not found at $A2_ADAPTER. Run: A2_ADAPTER=/path/to/qwen_4b_hinglish_lora bash submit_all_a3.sh"
  exit 1
fi

jobid() { echo "${1%%;*}"; }
LAST=""

submit() {
  local deps="$1"; shift
  if [ "$SEQUENTIAL" = "1" ] && [ -n "$LAST" ] && [[ ",$deps," != *":$LAST,"* ]]; then
    deps="${deps:+$deps,}afterany:$LAST"
  fi
  if [ -n "$deps" ]; then
    JOB=$(jobid "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=$deps "$@")")
  else
    JOB=$(jobid "$(sbatch --parsable "$@")")
  fi
  if ! [[ "$JOB" =~ ^[0-9]+$ ]]; then
    echo "Submission failed for: $*"
    echo "Nothing after this was submitted. Fix the error above, check squeue -u \$USER, cancel any jobs listed, and run again."
    exit 1
  fi
  LAST="$JOB"
}

PREP_DEP=""
if [ -f data_a3/test_indictalk.csv ] && [ "${SKIP_PREPARE:-0}" = "1" ]; then
  echo "Using existing data_a3/"
else
  submit "" prepare_data_a3.sh
  PREP=$JOB
  PREP_DEP="afterok:$PREP"
  echo "prepare data            job $PREP"
fi

A2F_DEP="$PREP_DEP"
if [[ " $RUNS " == *" E "* || " $RUNS " == *" F "* ]]; then
  if [ -f data_a3/train_a2_full.csv ] && [ "${SKIP_PREPARE:-0}" = "1" ]; then
    echo "Using existing data_a3/train_a2_full.csv"
  else
    submit "$PREP_DEP" prepare_a2_full_a3.sh
    A2F_DEP="afterok:$JOB"
    echo "prepare A2 full data    job $JOB"
  fi
fi

A2ALL_DEP="$PREP_DEP"
if [[ " $RUNS " == *" G "* ]]; then
  if [ -f data_a3/train_a2_all.csv ] && [ "${SKIP_PREPARE:-0}" = "1" ]; then
    echo "Using existing data_a3/train_a2_all.csv"
  else
    submit "$PREP_DEP" --export=ALL,OUT_FILE=train_a2_all.csv,MAX_A2_TRAIN=0 prepare_a2_full_a3.sh
    A2ALL_DEP="afterok:$JOB"
    echo "prepare A2 all data     job $JOB"
  fi
fi

for RUN in $RUNS; do
  if [ "$RUN" = "G" ]; then
    submit "$A2ALL_DEP" --export=ALL,RUN=G1 train_4b_a3.sh
    G1=$JOB
    echo "train G1                job $G1"
    submit "afterok:$G1" --export=ALL,RUN=G1 evaluate_a3.sh
    echo "evaluate G1             job $JOB"
    submit "afterok:$G1" --export=ALL,RUN=G2 train_4b_a3.sh
    G2=$JOB
    echo "train G2                job $G2"
    submit "afterok:$G2" --export=ALL,RUN=G2 evaluate_a3.sh
    echo "evaluate G2             job $JOB"
    continue
  fi
  TRAIN_DEP="$PREP_DEP"
  if [ "$RUN" = "E" ] || [ "$RUN" = "F" ]; then
    TRAIN_DEP="$A2F_DEP"
  fi
  submit "$TRAIN_DEP" --export=ALL,RUN=$RUN train_4b_a3.sh
  T=$JOB
  echo "train $RUN                 job $T"
  submit "afterok:$T" --export=ALL,RUN=$RUN evaluate_a3.sh
  E=$JOB
  echo "evaluate $RUN              job $E"
  if [ "$RUN" = "D" ]; then
    submit "afterok:$T" --export=ALL,RUN=D,USE_PERSONA=0 evaluate_a3.sh
    E=$JOB
    echo "evaluate D no persona    job $E"
  fi
  if [ "$RUN" = "D" ] && [ "$BASELINES" = "1" ]; then
    submit "$PREP_DEP" --export=ALL,RUN=A evaluate_a3.sh
    E=$JOB
    echo "evaluate A               job $E"
    submit "$PREP_DEP" --export=ALL,RUN=BASE evaluate_a3.sh
    E=$JOB
    echo "evaluate BASE            job $E"
  fi
done

if [[ " $RUNS " != *" D "* ]] && [ "$BASELINES" = "1" ]; then
  submit "$PREP_DEP" --export=ALL,RUN=A evaluate_a3.sh
  E=$JOB
  echo "evaluate A               job $E"
  submit "$PREP_DEP" --export=ALL,RUN=BASE evaluate_a3.sh
  E=$JOB
  echo "evaluate BASE            job $E"
fi

echo
if [ "$SEQUENTIAL" = "1" ]; then
  echo "Jobs run one after another (one GPU at a time). Use SEQUENTIAL=0 to run independent jobs in parallel."
fi
echo "Check progress with: squeue -u \$USER"
echo "Results table: results/results_table.md (updated after each evaluation)"
