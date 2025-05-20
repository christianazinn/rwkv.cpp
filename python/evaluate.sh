#!/bin/bash

export TEMPERATURE_SAMPLING=1.0
export REPETITION_PENALTY=1.2
export TOP_K=20
export TOP_P=0.95

export MODEL="mrwkv"
# run train/convert_model_to_cpp.sh if this doesn't yet exist
export MODEL_PATH="$PROJECT_ROOT/rwkv.cpp/python/rwkv_cpp/rcpp.bin"
export STATE_PATH="$PROJECT_ROOT/RWKV-PEFT/peft_model/rwkv-16.pth"

# inference parameters
export N_BARS=2
export CTX=8 # 4 * N_BARS usually
export num_generations=500
export DRUMS=0
export END_INFILLING=1
export MAX_NEW_TOKENS=2000

# pop909 does pop909 test set, must UNSET instead of set to 0 for gigamidi test
export pop909=1
# tests MIDI-Mistral comparison model
export mistral=0
# for end infillings on partial data, i.e. not actually the end of the song
# but any window in the song where we only use left context
export partial_end=1

python3 generate.py \
    --num_bars_infilling $N_BARS \
    --context $CTX \
    --num_generations $num_generations \
    --drums $DRUMS \
    --end_infilling $END_INFILLING \
    --mistral $mistral

# this script will automatically create the output folder and run objective evals
destination_folder="$PROJECT_ROOT/MIDIMetrics/tests/FINALTEST/${MODEL}/bars_infill${N_BARS}_context${CTX}"
if [ "$pop909" = "1" ]; then
   destination_folder="${destination_folder}_pop909"
fi
rm -r $destination_folder
mv output/* $destination_folder

curl -d "job done" ntfy.sh/your_ntfy_topic_here

cd "$PROJECT_ROOT/MIDIMetrics"
python3 -m tests.test_metrics
