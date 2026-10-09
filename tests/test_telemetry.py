"""Telemetry parsers (DJI SRT, CSV) + interpolation, via the parse_telemetry dispatch."""
import pytest

from drishti.config import CSVMapping, TelemetrySpec
from drishti.io.telemetry import parse_telemetry
from drishti.io.telemetry.base import TelemetryError

SRT_NEW = """1
00:00:00,000 --> 00:00:00,033
[latitude: 22.5431] [longitude: 113.9425] [rel_alt: 0.0 abs_alt: 54.4] [yaw: 10.0]

2
00:00:01,000 --> 00:00:01,033
[latitude: 22.5432] [longitude: 113.9426] [rel_alt: 1.0 abs_alt: 55.4] [yaw: 12.0]
"""

SRT_OLD = """1
00:00:00,000 --> 00:00:00,033
GPS(113.9425,22.5431,20) BAROMETER:54.4
"""


def test_parse_dji_srt_new_layout(tmp_path):
    p = tmp_path / "f.SRT"
    p.write_text(SRT_NEW, encoding="utf-8")
    tel = parse_telemetry(TelemetrySpec(format="dji_srt", path=str(p)))
    assert len(tel) == 2
    assert tel.samples[0].t == 0.0
    assert tel.samples[1].t == pytest.approx(1.0)
    assert tel.samples[0].lat == 22.5431
    assert tel.samples[0].lon == 113.9425
    assert tel.samples[0].alt == 54.4
    assert {"yaw", "rel_alt"}.issubset(tel.fields_present)


def test_parse_dji_srt_old_gps_layout(tmp_path):
    p = tmp_path / "o.SRT"
    p.write_text(SRT_OLD, encoding="utf-8")
    tel = parse_telemetry(TelemetrySpec(format="dji_srt", path=str(p)))
    assert tel.samples[0].lon == 113.9425
    assert tel.samples[0].lat == 22.5431
    assert tel.samples[0].alt == 54.4          # BAROMETER overrides the GPS altitude


def test_parse_csv_and_interpolate(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(
        "timestamp,latitude,longitude,abs_alt\n0.0,22.5,113.9,54.0\n1.0,22.6,114.0,55.0\n",
        encoding="utf-8",
    )
    tel = parse_telemetry(TelemetrySpec(format="csv", path=str(p), csv=CSVMapping()))
    assert len(tel) == 2
    assert tel.samples[0].lat == 22.5
    mid = tel.sample_at(0.5)
    assert mid.lat == pytest.approx(22.55)
    assert mid.lon == pytest.approx(113.95)
    assert mid.alt == pytest.approx(54.5)


def test_sample_at_clamps_to_range(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text(
        "timestamp,latitude,longitude,abs_alt\n0.0,10,20,0\n2.0,12,22,0\n", encoding="utf-8"
    )
    tel = parse_telemetry(TelemetrySpec(format="csv", path=str(p)))
    assert tel.sample_at(-5.0).lat == 10        # clamped low
    assert tel.sample_at(99.0).lat == 12        # clamped high


def test_missing_file_raises():
    with pytest.raises(TelemetryError):
        parse_telemetry(TelemetrySpec(format="csv", path="does_not_exist.csv"))


def test_no_path_raises():
    with pytest.raises(TelemetryError):
        parse_telemetry(TelemetrySpec(format="dji_srt", path=None))


def test_srt_without_gps_raises(tmp_path):
    p = tmp_path / "empty.SRT"
    p.write_text("1\n00:00:00,000 --> 00:00:00,033\nno gps here\n", encoding="utf-8")
    with pytest.raises(TelemetryError):
        parse_telemetry(TelemetrySpec(format="dji_srt", path=str(p)))
