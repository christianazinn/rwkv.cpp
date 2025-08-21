import sys
import os
from music_json_convert import json_prompt_to_midi


if __name__ == "__main__":
    prompt_json_fn = sys.argv[1]
    # directory of this file
    output_midi_folder = os.path.dirname(__file__)

    os.makedirs(output_midi_folder, exist_ok=True)

    # Convert the prompt_json to midi
    json_prompt_to_midi(prompt_json_fn, os.path.join(output_midi_folder, 'prompt.mid'), bpm=90, vel=80)