import pretty_midi as pm
import numpy as np
import json


def note_matrix_to_json_dict(note_matrix, key_name):
    notes = []
    for row in note_matrix:
        start, pitch, duration = row
        notes.append({
            "start": int(start * 4),
            "pitch": int(pitch),
            "duration": int(duration * 4)
        })
    return {key_name: notes}


def write_json_dict(json_dict, output_json_fn):
    with open(output_json_fn, "w") as f:
        json.dump(json_dict, f, indent=2)


def json_dict_to_note_matrix(json_dict, key_name):
    notes = json_dict[key_name]
    note_matrix = []
    for note in notes:
        start = note["start"] / 4.0
        pitch = note["pitch"]
        duration = note["duration"] / 4.0
        note_matrix.append([start, pitch, duration])
    return np.array(note_matrix)


def read_json_dict(input_json_fn):
    with open(input_json_fn, "r") as f:
        json_dict = json.load(f)
    return json_dict


def midi_to_note_matrix(input_midi_fn):
    midi = pm.PrettyMIDI(input_midi_fn)

    tempo_changes = midi.get_tempo_changes()
    if len(tempo_changes[1]) != 1:
        raise ValueError("Supports MIDI files having exactly one tempo change.")
    bpm = tempo_changes[1][0]

    notes = [note for ins in midi.instruments for note in ins.notes]

    note_matrix = []

    for note in notes:
        start = note.start * bpm / 60
        end = note.end * bpm / 60
        start = np.round(start * 4) / 4
        duration = max(np.round((end - start) * 4) / 4, 0.25)
        note_matrix.append([start, note.pitch, duration])
    note_matrix = np.array(note_matrix)
    note_matrix = note_matrix[note_matrix[:, 0].argsort()]
    return note_matrix


def note_matrix_to_midi(note_matrix, output_midi_fn, bpm=90, vel=80):
    midi = pm.PrettyMIDI(initial_tempo=bpm)
    instrument = pm.Instrument(program=0) 

    for note in note_matrix:
        start, pitch, duration = note
        start_time = float(start) * 60 / bpm
        end_time = float(start + duration) * 60 / bpm
        note = pm.Note(velocity=int(vel), pitch=int(pitch), start=start_time, end=end_time)
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(output_midi_fn)


def midi_prompt_to_json(input_midi_fn, output_json_fn):
    note_matrix = midi_to_note_matrix(input_midi_fn)
    json_dict = note_matrix_to_json_dict(note_matrix, 'prompt')
    write_json_dict(json_dict, output_json_fn)


def json_prompt_to_midi(input_json_fn, output_midi_fn, bpm=90, vel=80):
    json_dict = read_json_dict(input_json_fn)
    note_matrix = json_dict_to_note_matrix(json_dict, 'prompt')
    note_matrix_to_midi(note_matrix, output_midi_fn, bpm=bpm, vel=vel)