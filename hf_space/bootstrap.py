"""Install pinned upstream adapters without their incompatible metadata graph."""
from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys

PACKAGES = (
    ('basic-pitch', '0.4.0', 'basic-pitch==0.4.0', None),
    ('piano-transcription-inference', None,
     'piano-transcription-inference@git+https://github.com/xavriley/piano_transcription_inference.git@7568dc7f78b625e40cf9776e2806d164006610e3',
     '7568dc7f78b625e40cf9776e2806d164006610e3'),
    ('hf-midi-transcription', None,
     'hf-midi-transcription@git+https://github.com/xavriley/hf_midi_transcription.git@96f6797881e9497cbfc8f8e5deccea9c1f2f7adc',
     '96f6797881e9497cbfc8f8e5deccea9c1f2f7adc'),
)


def install_adapters():
    for name, version, requirement, commit in PACKAGES:
        try:
            distribution = importlib.metadata.distribution(name)
            if commit:
                installed = json.loads(distribution.read_text('direct_url.json') or '{}').get('vcs_info', {}).get('commit_id')
                ready = installed == commit
            else:
                ready = distribution.version == version
        except importlib.metadata.PackageNotFoundError:
            ready = False
        if not ready:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-input', '--no-deps', requirement], check=True, timeout=300)
