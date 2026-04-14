"""
Tests for src/export/exporter.py
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.export.exporter import export_json, export_csv, _make_serialisable


class TestMakeSerializable:
    def test_dict_passthrough(self):
        result = _make_serialisable({"a": 1, "b": "text"})
        assert result == {"a": 1, "b": "text"}

    def test_numpy_int(self):
        result = _make_serialisable(np.int64(42))
        assert isinstance(result, int)

    def test_numpy_float(self):
        result = _make_serialisable(np.float32(3.14))
        assert isinstance(result, float)

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = _make_serialisable(arr)
        assert isinstance(result, list)

    def test_dataframe_becomes_records(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _make_serialisable(df)
        assert isinstance(result, list)
        assert result[0] == {"a": 1, "b": 3}

    def test_series_becomes_dict(self):
        s = pd.Series({"x": 10, "y": 20})
        result = _make_serialisable(s)
        assert isinstance(result, dict)

    def test_nested_structure(self):
        nested = {"metrics": {"r2": np.float64(0.95), "data": pd.Series([1, 2])}}
        result = _make_serialisable(nested)
        assert isinstance(result["metrics"]["r2"], float)
        assert isinstance(result["metrics"]["data"], dict)


class TestExportJson:
    def test_file_created(self, tmp_path):
        data = {"mean": 100.0, "p5": 80.0}
        path = export_json(data, output_dir=str(tmp_path), timestamp=False)
        assert path.exists()

    def test_file_is_valid_json(self, tmp_path):
        data = {"value": 42, "name": "test"}
        path = export_json(data, output_dir=str(tmp_path), timestamp=False)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["value"] == 42

    def test_timestamp_prefix(self, tmp_path):
        path = export_json({}, output_dir=str(tmp_path), timestamp=True)
        assert len(path.stem) > 10  # timestamp prefix adds characters

    def test_dataframe_serialised(self, tmp_path):
        df = pd.DataFrame({"channel": ["tv", "radio"], "spend": [1000, 2000]})
        path = export_json({"allocation": df}, output_dir=str(tmp_path), timestamp=False)
        with open(path) as f:
            loaded = json.load(f)
        assert isinstance(loaded["allocation"], list)


class TestExportCsv:
    def test_file_created(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        path = export_csv(df, "test.csv", output_dir=str(tmp_path), timestamp=False)
        assert path.exists()

    def test_roundtrip(self, tmp_path):
        df = pd.DataFrame({"channel": ["tv", "radio"], "spend": [1000, 2000]})
        path = export_csv(df, "out.csv", output_dir=str(tmp_path), timestamp=False)
        loaded = pd.read_csv(path)
        pd.testing.assert_frame_equal(df, loaded)
