#!/bin/bash

export TEMPERATURE_SAMPLING=1.0
export REPETITION_PENALTY=1.2
export TOP_K=20
export TOP_P=0.95

export MODEL="mrwkv"
# rcpj should be the base one for rwkv-12, idk
export MODEL_PATH="/home/christian/MIDI-RWKV/src/outputs/m2fla/rcpp.bin"
export STATE_PATH="" # "/home/christian/MIDI-RWKV/RWKV-PEFT/peft_model/rwkv-6.pth" # ""

export N_BARS=2
export CTX=8
num_generations=200
export DRUMS=0
export END_INFILLING=0
export beatles=1
export mistral=0

python3 generate.py \
    --num_bars_infilling $N_BARS \
    --context $CTX \
    --num_generations $num_generations \
    --drums $DRUMS \
    --end_infilling $END_INFILLING \
    --mistral $mistral

destination_folder="/home/christian/MIDI-RWKV/MIDIMetrics/tests/FINALTEST/${MODEL}/bars_infill${N_BARS}_context${CTX}"
if [ "$beatles" = "1" ]; then
  destination_folder="${destination_folder}_beatles"
fi
rm -r $destination_folder
mv output/* $destination_folder

curl -d "job done" ntfy.sh/phantasmagoria

cd "/home/christian/MIDI-RWKV/MIDIMetrics"
python3 -m tests.test_metrics
