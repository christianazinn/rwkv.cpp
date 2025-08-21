from music_json_convert import midi_prompt_to_json
import os
import sys


def convert_all_midi_files_to_json(raw_midi_folder, output_json_folder):
    os.makedirs(output_json_folder, exist_ok=True)

    for filename in os.listdir(raw_midi_folder):
        if filename.endswith('.mid'):
            input_midi_fn = os.path.join(raw_midi_folder, filename)
            output_json_fn = os.path.join(output_json_folder, filename.replace('.mid', '.json'))
            midi_prompt_to_json(input_midi_fn, output_json_fn)
            print(f"Converted {filename} to JSON format.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prompt_midi_to_json.py <raw_midi_folder> <output_json_folder>")
        sys.exit(1)

    raw_midi_folder = sys.argv[1]
    output_json_folder = sys.argv[2]

    convert_all_midi_files_to_json(raw_midi_folder, output_json_folder)