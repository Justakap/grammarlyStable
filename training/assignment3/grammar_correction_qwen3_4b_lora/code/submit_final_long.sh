#!/bin/bash
set -e
cd "$(dirname "$0")"

RUNS=${RUNS:-"G2_long 8B_long"}
LONG_DATA=${LONG_DATA:-data_a3_full}
COMPARE_RUNS=${COMPARE_RUNS:-G2}
DATA_SETTINGS="MAX_WORDS=${MAX_WORDS:-120},MAX_INDICTALK_TRAIN=${MAX_INDICTALK_TRAIN:-0},GRAMMAR_SCALE_WORDS=${GRAMMAR_SCALE_WORDS:-40},KEEP_SPLIT_FROM=data_a3"

for R in $RUNS; do
  case "$R" in
    G2_long|8B_long) ;;
    *) echo "Unknown run $R in RUNS. Use G2_long and/or 8B_long."; exit 1 ;;
  esac
  if [ -d "runs/run_$R" ]; then
    echo "runs/run_$R already exists. Remove or rename it before training $R again."
    exit 1
  fi
done
if [[ " $RUNS " == *" G2_long "* ]] && [ ! -f runs/run_G1/best/adapter_config.json ]; then
  echo "runs/run_G1/best not found. G2_long continues from run G1."
  exit 1
fi
if [ ! -f data_a3/train_indictalk.csv ] || [ ! -f data_a3/test_indictalk.csv ]; then
  echo "data_a3/ not found. Run prepare_data_a3.sh first."
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

if [ -f runs/run_G1/train_summary.json ] && [ "${SKIP_TIME_CHECK:-0}" != "1" ]; then
  set +e
  RUNS="$RUNS" TRAIN_TIME_4B=$TRAIN_TIME_4B TRAIN_TIME_8B=$TRAIN_TIME_8B \
    MAX_TRAIN_EXAMPLES_8B=${MAX_TRAIN_EXAMPLES_8B:-0} python3 - <<'EOF'
import json, os, sys
m = json.load(open("runs/run_G1/train_summary.json")).get("training_minutes")
if not m:
    sys.exit(0)
g1 = float(m) / 60
print(f"G1 trained for {g1:.1f} hours.")
cap = int(os.environ["MAX_TRAIN_EXAMPLES_8B"] or 0)
stop = False
for run, factor, time_var, share in (("G2_long", 1.2, "TRAIN_TIME_4B", 1.0),
                                     ("8B_long", 2.6, "TRAIN_TIME_8B", min(1.0, cap / 417000) if cap > 0 else 1.0)):
    if run not in os.environ["RUNS"].split():
        continue
    est = factor * g1 * share
    print(f"  {run} should take about {est:.0f} hours (job limit 48 unless {time_var} is set).")
    if est > 44 and not os.environ[time_var]:
        stop = True
        fix = f"{time_var}=3-00:00:00 (if the partition allows it: sinfo -p gpu -o '%P %l')"
        if run == "8B_long":
            fix += f", or MAX_TRAIN_EXAMPLES_8B={int(417000 * 40 / (factor * g1) / 1000) * 1000} to fit in about 40 hours"
        print(f"  That may not finish in time. Rerun with {fix}.")
sys.exit(3 if stop else 0)
EOF
  status=$?
  set -e
  if [ "$status" = "3" ]; then
    echo "Nothing was submitted. (SKIP_TIME_CHECK=1 submits anyway.)"
    exit 1
  fi
fi

if [ -f "$LONG_DATA/test_indictalk.csv" ] && [ -f "$LONG_DATA/train_a2_replay.csv" ]; then
  echo "Using existing $LONG_DATA/"
  grep -m1 "Settings" "$LONG_DATA/prepare_data_a3.log" 2>/dev/null || true
  DATA_DEP=""
else
  submit "" --export=ALL,DATA_DIR=$LONG_DATA,$DATA_SETTINGS prepare_data_a3.sh
  DATA_DEP="afterok:$JOB"
  echo "Data step: job $JOB builds $LONG_DATA/ ($DATA_SETTINGS)"
fi

TRAINED=""
for R in $RUNS; do
  if [ "$R" = "G2_long" ]; then
    EXTRA="${BATCH_SIZE_4B:+,BATCH_SIZE=$BATCH_SIZE_4B}${GRAD_ACCUM_STEPS_4B:+,GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS_4B}"
    TIME=$TRAIN_TIME_4B
  else
    EXTRA="${BATCH_SIZE_8B:+,BATCH_SIZE=$BATCH_SIZE_8B}${GRAD_ACCUM_STEPS_8B:+,GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS_8B}${MAX_TRAIN_EXAMPLES_8B:+,MAX_TRAIN_EXAMPLES=$MAX_TRAIN_EXAMPLES_8B}"
    TIME=$TRAIN_TIME_8B
  fi
  submit "$DATA_DEP" ${TIME:+--time=$TIME} --export=ALL,RUN=$R,LONG_DATA=$LONG_DATA$EXTRA train_4b_a3.sh
  TRAINED="$TRAINED $R:$JOB"
  echo "$R: train job $JOB"
done

for P in $TRAINED; do
  R=${P%%:*}
  T=${P##*:}
  submit "afterok:$T" --export=ALL,RUN=$R,LONG_DATA=$LONG_DATA,TEST_SET=original evaluate_a3.sh
  echo "  $R on the original test sets: job $JOB -> results/results_table.md (next to runs A-G)"
  submit "afterok:$T" --export=ALL,RUN=$R,LONG_DATA=$LONG_DATA,TEST_SET=long evaluate_a3.sh
  echo "  $R on the long-message test set: job $JOB -> results_long/results_table.md"
done

for R in $COMPARE_RUNS; do
  if [ -f "runs/run_$R/train_summary.json" ]; then
    submit "$DATA_DEP" --export=ALL,RUN=$R,LONG_DATA=$LONG_DATA,TEST_SET=long evaluate_a3.sh
    echo "  $R on the long-message test set, for comparison: job $JOB -> results_long/results_table.md"
  else
    echo "  runs/run_$R not found, so $R is not scored on the long-message test set"
  fi
done

echo
echo "Check progress with: squeue -u \$USER"
echo "Results: results/results_table.md (original test sets, next to runs A-G) and results_long/results_table.md (long messages)"
