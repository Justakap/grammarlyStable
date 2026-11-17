#!/bin/bash
set -e
cd "$(dirname "$0")"

SIZES=${SIZES:-"0.6B"}
BASELINES=${BASELINES:-1}
DATA_DIR=${DATA_DIR:-data_nwp}

for S in $SIZES; do
  case "$S" in
    0.6B|1.7B) ;;
    *) echo "Unknown size $S in SIZES. Use 0.6B and/or 1.7B."; exit 1 ;;
  esac
  if [ -d "runs/run_$S" ]; then
    echo "runs/run_$S already exists. Remove or rename it before training $S again."
    exit 1
  fi
done
if [ -n "$A2_MODEL" ] && [ ! -f "$A2_MODEL/config.json" ]; then
  echo "A2_MODEL=$A2_MODEL has no config.json. Point it at the Assignment 2 autocomplete model folder."
  exit 1
fi

jobid() { echo "${1%%;*}"; }
submit() {
  local deps="$1"; shift
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
}

if [ -f "$DATA_DIR/train.jsonl" ] && [ -f "$DATA_DIR/test_a2.jsonl" ]; then
  echo "Using existing $DATA_DIR/"
  DATA_DEP=""
else
  submit "" --export=ALL,DATA_DIR=$DATA_DIR prepare_data_nwp.sh
  DATA_DEP="afterok:$JOB"
  echo "Data step: job $JOB builds $DATA_DIR/"
fi

TRAINED=""
for S in $SIZES; do
  submit "$DATA_DEP" --export=ALL,SIZE=$S,DATA_DIR=$DATA_DIR train_next_word.sh
  TRAINED="$TRAINED $S:$JOB"
  echo "$S: train job $JOB"
done

IT_TEST="$DATA_DIR/test_indictalk.jsonl"
for P in $TRAINED; do
  S=${P%%:*}
  T=${P##*:}
  submit "afterok:$T" --export=ALL,RUN=$S,PERSONA_MODE=true,DATA_DIR=$DATA_DIR evaluate_next_word.sh
  echo "  $S with the true persona (both test sets): job $JOB"
  submit "afterok:$T" --export=ALL,RUN=$S,PERSONA_MODE=none,DATA_DIR=$DATA_DIR,TEST_FILES=$IT_TEST evaluate_next_word.sh
  echo "  $S with persona 'none' (IndicTalk test): job $JOB"
  submit "afterok:$T" --export=ALL,RUN=$S,PERSONA_MODE=wrong,DATA_DIR=$DATA_DIR,TEST_FILES=$IT_TEST evaluate_next_word.sh
  echo "  $S with a wrong persona (IndicTalk test): job $JOB"
done

if [ "$BASELINES" = "1" ]; then
  submit "$DATA_DEP" --export=ALL,RUN=BASE,DATA_DIR=$DATA_DIR evaluate_next_word.sh
  echo "BASE (Qwen3-0.6B-Base, no fine-tuning): job $JOB"
  if [ -n "$A2_MODEL" ]; then
    submit "$DATA_DEP" --export=ALL,RUN=A2,A2_MODEL=$A2_MODEL,DATA_DIR=$DATA_DIR evaluate_next_word.sh
    echo "A2 (Assignment 2 autocomplete model): job $JOB"
  else
    echo "A2_MODEL not set, so the Assignment 2 model is not evaluated"
  fi
fi

echo
echo "Check progress with: squeue -u \$USER"
echo "Results: results/results_table.md"
