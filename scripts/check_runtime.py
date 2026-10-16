"""Check the installed frozen runtime, with two explicit upstream metadata exceptions."""
import importlib.metadata as metadata
import sys
import json
from pathlib import Path

from packaging.requirements import Requirement

errors = []
for distribution in metadata.distributions():
    for raw in distribution.requires or []:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate({'extra':''}):
            continue
        # Fretwork uses Basic Pitch's bundled ONNX model. TensorFlow is unused;
        # resampy 0.4.3 is required by the guitar adapter and tested end-to-end.
        if distribution.metadata['Name'].lower() == 'basic-pitch' and requirement.name.lower() in {'tensorflow','resampy'}:
            continue
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            errors.append(f'{distribution.metadata["Name"]} needs {requirement}')
        else:
            if requirement.specifier and version not in requirement.specifier:
                errors.append(f'{distribution.metadata["Name"]} needs {requirement}; found {version}')
if errors:
    raise SystemExit('\n'.join(errors))
for name, expected in {
    'hf-midi-transcription':'96f6797881e9497cbfc8f8e5deccea9c1f2f7adc',
    'piano-transcription-inference':'7568dc7f78b625e40cf9776e2806d164006610e3',
}.items():
    source = json.loads(metadata.distribution(name).read_text('direct_url.json') or '{}')
    assert source.get('vcs_info',{}).get('commit_id') == expected, f'{name} must use its locked upstream commit'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp.server import MCPServer
from basic_pitch.inference import Model
from basic_pitch import ICASSP_2022_MODEL_PATH
from hf_midi_transcription import MidiTranscriptionModel
from fastapi import FastAPI
model = Model(ICASSP_2022_MODEL_PATH)
assert str(model.model_type).lower().endswith('onnx'), 'The supported reviewer runtime is ONNX, not TensorFlow'
print('Fretwork runtime dependency closure, MCP v2, ONNX reviewer and API imports passed')
