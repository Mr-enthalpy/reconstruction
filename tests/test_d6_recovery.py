import json
from pathlib import Path
import pytest
from experiments.d6.recover_summary import recover


def test_saved_rows_recover_without_cube_access():
    root=Path(__file__).resolve().parents[1]/'docs/reports/2026-11-01-d6'
    original=(root/'g4_results.json').read_text(encoding='utf-8')
    protocol=json.loads((root/'g4_protocol.json').read_bytes())
    expected=json.loads((root/'g4_results_recovered.json').read_bytes())
    assert recover(original,protocol)==expected
    assert recover(original.replace('\n','\r\n'),protocol)==expected
    with pytest.raises(ValueError):recover(original[:1000],protocol)
