import sys
import tomllib

from pathlib import Path
from typing import Any

import pytest

from biahub.dca_metadata import (
    AIRTABLE_FIELDS,
    TABLE_COLUMNS,
    PyairtableClient,
    airtable_rows,
    blank_row,
    metadata_rows,
    read_metadata_table,
)
from biahub.settings import AirtableSettings, DcaFinalizeSettings

REPO = Path(__file__).parents[1]
EXAMPLE_SHEET = REPO / "settings" / "example_dca_metadata.csv"
A549_BASE = REPO / "nextflow" / "configs" / "a549" / "dca_base.toml"

ROW_C2 = ["ds", "C/2", "A549", "TOMM20", "mitochondria", "ZIKV", "5", "30", "24"]
ROW_B1 = ["ds", "B/1", "A549", "SEC61B", "endoplasmic_reticulum", "mock", "0", "30", "24"]


def write_table(path: Path, rows: list[list[str]], sep: str, columns=TABLE_COLUMNS) -> Path:
    lines = [sep.join(columns)] + [sep.join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def pyairtable_record(well: str, **overrides: Any) -> dict[str, Any]:
    """One record in the lab table's shape: ``cell_line`` is a linked-record field."""
    fields = {
        "dataset": "ds",
        "well_id": well,
        "cell_line": ["recCELLLINE000001", "recCELLLINE000002"],
        "cell_type": "A549",
        "marker": "TOMM20",
        "perturbation": "ZIKV",
        "moi": 5,
        "time_interval_min": 30,
        "hours_post_perturbation": 24.5,
    }
    fields.update(overrides)
    return {
        "id": "recXXXXXXXXXXXXXX",
        "createdTime": "2024-01-01T00:00:00.000Z",
        "fields": fields,
    }


def airtable_settings(**overrides: Any) -> AirtableSettings:
    values = {
        "base_id": "b",
        "table": "t",
        "dataset": "ds",
        "values": {"organelle": "mitochondria"},
    }
    return AirtableSettings(**(values | overrides))


class FakeClient:
    def __init__(self, records: list[dict[str, Any]]):
        self._records = records
        self.calls: list[str] = []

    def records(self, dataset: str) -> list[dict[str, Any]]:
        self.calls.append(dataset)
        return self._records


def test_read_csv_and_tsv(tmp_path):
    rows = [ROW_C2, ROW_B1]
    csv = read_metadata_table(write_table(tmp_path / "t.csv", rows, ","))
    tsv = read_metadata_table(write_table(tmp_path / "t.tsv", rows, "\t"))
    assert csv == tsv
    assert set(csv) == {"C/2", "B/1"}
    assert csv["C/2"]["marker"] == "TOMM20"
    assert csv["B/1"]["moi"] == "0"


def test_duplicate_well_raises(tmp_path):
    path = write_table(tmp_path / "t.csv", [ROW_C2, ROW_C2], ",")
    with pytest.raises(ValueError, match="C/2"):
        read_metadata_table(path)


def test_missing_column_raises(tmp_path):
    columns = tuple(c for c in TABLE_COLUMNS if c != "marker")
    rows = [[v for c, v in zip(TABLE_COLUMNS, ROW_C2, strict=True) if c != "marker"]]
    path = write_table(tmp_path / "t.csv", rows, ",", columns)
    with pytest.raises(ValueError, match="marker"):
        read_metadata_table(path)


def test_blank_row_has_every_column():
    assert blank_row() == {c: "" for c in TABLE_COLUMNS}


def test_airtable_fields_cover_every_column():
    assert set(AIRTABLE_FIELDS) == set(TABLE_COLUMNS)


def test_airtable_rows_map_pyairtable_record_shape():
    client = FakeClient([pyairtable_record("C/2")])
    rows = airtable_rows(client, airtable_settings())
    assert client.calls == ["ds"]
    assert rows["C/2"] == {
        "dataset": "ds",
        "well": "C/2",
        "cell_line": "A549",
        "marker": "TOMM20",
        "organelle": "mitochondria",
        "perturbation": "ZIKV",
        "moi": "5",
        "time_interval_min": "30",
        "hours_post_perturbation": "24.5",
    }


def test_airtable_missing_field_gives_empty_string():
    record = pyairtable_record("C/2")
    del record["fields"][AIRTABLE_FIELDS["moi"]]
    rows = airtable_rows(FakeClient([record]), airtable_settings())
    assert rows["C/2"]["moi"] == ""


def test_airtable_multi_select_cell_line_joined():
    record = pyairtable_record("C/2", **{AIRTABLE_FIELDS["cell_line"]: ["A549", "HEK293T"]})
    rows = airtable_rows(FakeClient([record]), airtable_settings())
    assert rows["C/2"]["cell_line"] == "A549,HEK293T"


def test_airtable_duplicate_template_raises():
    client = FakeClient([pyairtable_record("C/2"), pyairtable_record("C/2")])
    with pytest.raises(ValueError, match="C/2"):
        airtable_rows(client, airtable_settings())


def test_airtable_record_without_well_raises():
    record = pyairtable_record("C/2")
    del record["fields"][AIRTABLE_FIELDS["well"]]
    with pytest.raises(ValueError, match="recXXXXXXXXXXXXXX"):
        airtable_rows(FakeClient([record]), airtable_settings())


def test_table_next_to_plate_wins_over_airtable(tmp_path):
    plate = tmp_path / "plate.zarr"
    write_table(tmp_path / "plate.dca-metadata.csv", [["plate", *ROW_C2[1:]]], ",")
    client = FakeClient([pyairtable_record("B/1")])
    settings = DcaFinalizeSettings(base_config="b.toml", airtable=airtable_settings())
    rows = metadata_rows(plate, settings, client)
    assert set(rows) == {"C/2"}
    assert client.calls == []


def test_airtable_used_when_no_table(tmp_path):
    plate = tmp_path / "plate.zarr"
    client = FakeClient([pyairtable_record("C/2"), pyairtable_record("B/1")])
    settings = DcaFinalizeSettings(base_config="b.toml", airtable=airtable_settings())
    rows = metadata_rows(plate, settings, client)
    assert set(rows) == {"C/2", "B/1"}
    assert client.calls == ["ds"]


def test_no_source_raises(tmp_path):
    plate = tmp_path / "plate.zarr"
    settings = DcaFinalizeSettings(base_config="b.toml")
    expected = tmp_path / "plate.dca-metadata.csv"
    with pytest.raises(FileNotFoundError, match=str(expected)):
        metadata_rows(plate, settings, None)


def test_default_client_missing_token_names_variable(monkeypatch):
    monkeypatch.delenv("BIAHUB_TEST_AIRTABLE_TOKEN", raising=False)
    settings = airtable_settings(token_env="BIAHUB_TEST_AIRTABLE_TOKEN")
    with pytest.raises(RuntimeError, match="BIAHUB_TEST_AIRTABLE_TOKEN"):
        PyairtableClient(settings)


def test_airtable_template_wins_over_fov_records():
    template = pyairtable_record("C/2", **{AIRTABLE_FIELDS["marker"]: "TOMM20"})
    fov = pyairtable_record("C/2", fov="000000", **{AIRTABLE_FIELDS["marker"]: "OTHER"})
    rows = airtable_rows(FakeClient([fov, template]), airtable_settings())
    assert rows["C/2"]["marker"] == "TOMM20"


def test_airtable_agreeing_fov_records_collapse():
    records = [pyairtable_record("C/2", fov="000000"), pyairtable_record("C/2", fov="000001")]
    rows = airtable_rows(FakeClient(records), airtable_settings())
    assert list(rows) == ["C/2"]
    assert rows["C/2"]["marker"] == "TOMM20"


def test_airtable_disagreeing_fov_records_raise():
    records = [
        pyairtable_record("C/2", fov="000000"),
        pyairtable_record("C/2", fov="000001", **{AIRTABLE_FIELDS["moi"]: 1}),
    ]
    with pytest.raises(ValueError, match="C/2.*moi"):
        airtable_rows(FakeClient(records), airtable_settings())


def test_blank_well_cell_raises_with_row_number(tmp_path):
    blank = ["ds", "", "A549", "TOMM20", "mitochondria", "ZIKV", "5", "30", "24"]
    path = write_table(tmp_path / "t.csv", [ROW_C2, blank], ",")
    with pytest.raises(ValueError, match="row 3"):
        read_metadata_table(path)


@pytest.fixture
def no_pyairtable(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyairtable", None)
    monkeypatch.setitem(sys.modules, "pyairtable.formulas", None)


def test_csv_path_works_and_default_client_explains_missing_extra(
    no_pyairtable, tmp_path, monkeypatch
):
    monkeypatch.setenv("BIAHUB_TEST_AIRTABLE_TOKEN", "x")
    plate = tmp_path / "plate.zarr"
    write_table(tmp_path / "plate.dca-metadata.csv", [["plate", *ROW_C2[1:]]], ",")
    settings = DcaFinalizeSettings(base_config="b.toml")
    assert set(metadata_rows(plate, settings)) == {"C/2"}
    with pytest.raises(ImportError, match=r"biahub\[dca\]"):
        PyairtableClient(airtable_settings(token_env="BIAHUB_TEST_AIRTABLE_TOKEN"))


def test_shared_sheet_rows_filtered_by_plate_stem(tmp_path):
    shared = write_table(
        tmp_path / "shared.csv",
        [
            ["plate_a", *ROW_C2[1:]],
            ["plate_b", "C/2", *["other"] * 7],
            ["", *ROW_B1[1:]],
        ],
        ",",
    )
    settings = DcaFinalizeSettings(base_config="b.toml", metadata_table=shared)
    rows_a = metadata_rows(tmp_path / "plate_a.zarr", settings)
    rows_b = metadata_rows(tmp_path / "plate_b.zarr", settings)
    assert set(rows_a) == {"C/2", "B/1"}
    assert rows_a["C/2"]["marker"] == "TOMM20"
    assert rows_b["C/2"]["marker"] == "other"
    assert metadata_rows(tmp_path / "plate_c.zarr", settings) == {"B/1": rows_a["B/1"]}


def test_sheet_without_matching_dataset_gives_no_rows(tmp_path):
    write_table(tmp_path / "plate.dca-metadata.csv", [ROW_C2], ",")
    settings = DcaFinalizeSettings(base_config="b.toml")
    assert metadata_rows(tmp_path / "plate.zarr", settings) == {}


def test_example_sheet_reads_and_matches_the_a549_base():
    rows = read_metadata_table(EXAMPLE_SHEET, dataset="2026_01_02_example_plate")
    assert EXAMPLE_SHEET.read_text().splitlines()[0] == ",".join(TABLE_COLUMNS)
    assert len(rows) == 3
    base = tomllib.loads(A549_BASE.read_text())
    control_map = base["dca"]["perturbation"]["control_class"]["map"]
    assert {row["perturbation"] for row in rows.values()} <= set(control_map)


def test_airtable_linked_cell_line_ids_are_not_read():
    rows = airtable_rows(FakeClient([pyairtable_record("C/2")]), airtable_settings())
    assert rows["C/2"]["cell_line"] == "A549"


def test_airtable_na_perturbation_is_uninfected():
    record = pyairtable_record("C/2", perturbation="N/A")
    del record["fields"]["moi"]
    row = airtable_rows(FakeClient([record]), airtable_settings())["C/2"]
    assert (row["perturbation"], row["moi"]) == ("uninfected", "")


def test_airtable_column_without_a_field_needs_a_value():
    settings = airtable_settings(values={})
    with pytest.raises(ValueError, match=r"airtable\.values\.organelle"):
        airtable_rows(FakeClient([pyairtable_record("C/2")]), settings)


def test_airtable_field_map_override():
    record = pyairtable_record("C/2", organelle_name="endoplasmic_reticulum")
    settings = airtable_settings(fields={"organelle": "organelle_name"}, values={})
    row = airtable_rows(FakeClient([record]), settings)["C/2"]
    assert row["organelle"] == "endoplasmic_reticulum"


def test_airtable_value_for_a_mapped_column_raises():
    settings = airtable_settings(values={"organelle": "mitochondria", "marker": "SEC61B"})
    with pytest.raises(ValueError, match="marker"):
        airtable_rows(FakeClient([pyairtable_record("C/2")]), settings)


def test_airtable_unknown_column_raises():
    settings = airtable_settings(fields={"organel": "organelle"})
    with pytest.raises(ValueError, match="organel"):
        airtable_rows(FakeClient([pyairtable_record("C/2")]), settings)


def test_airtable_key_column_needs_a_field():
    settings = airtable_settings(
        fields={"well": None}, values={"organelle": "mitochondria", "well": "C/2"}
    )
    with pytest.raises(ValueError, match="well"):
        airtable_rows(FakeClient([pyairtable_record("C/2")]), settings)


def test_airtable_no_records_names_the_dataset():
    with pytest.raises(ValueError, match=r"dataset = 'ds'.*airtable\.dataset"):
        airtable_rows(FakeClient([]), airtable_settings())
