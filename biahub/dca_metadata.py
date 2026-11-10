"""Per-well infectomics metadata for ``dca-finalize``, from a sample sheet or AirTable."""

import os

from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from biahub.settings import AirtableSettings, DcaFinalizeSettings, dca_dataset_stem

TABLE_COLUMNS: tuple[str, ...] = (
    "dataset",
    "well",
    "cell_line",
    "marker",
    "organelle",
    "perturbation",
    "moi",
    "time_interval_min",
    "hours_post_perturbation",
)

AIRTABLE_FOV_FIELD = "fov"

AIRTABLE_FIELDS: dict[str, str | None] = {
    "dataset": "dataset",
    "well": "well_id",
    "cell_line": "cell_type",
    "marker": "marker",
    "organelle": None,
    "perturbation": "perturbation",
    "moi": "moi",
    "time_interval_min": "time_interval_min",
    "hours_post_perturbation": "hours_post_perturbation",
}


AIRTABLE_KEY_COLUMNS: tuple[str, ...] = ("dataset", "well")
AIRTABLE_PERTURBATION_ALIASES: dict[str, str] = {"N/A": "uninfected"}


def airtable_fields(settings: AirtableSettings) -> dict[str, str | None]:
    """Return the AirTable field each sample-sheet column reads, after the settings' overrides.

    A column with no field (``None``) takes its value from ``settings.values``.
    """
    configured = set(settings.fields) | set(settings.values)
    unknown = sorted(configured - set(TABLE_COLUMNS))
    if unknown:
        raise ValueError(f"airtable settings name unknown column(s): {', '.join(unknown)}")
    fields = AIRTABLE_FIELDS | settings.fields
    keys = [column for column in AIRTABLE_KEY_COLUMNS if fields[column] is None]
    if keys:
        raise ValueError(f"airtable column(s) {', '.join(keys)} must read an AirTable field")
    both = sorted(column for column in settings.values if fields[column] is not None)
    if both:
        raise ValueError(
            f"airtable.values sets {', '.join(both)}, which also read an AirTable field; "
            "map the field to null in airtable.fields to use the value"
        )
    unset = [c for c, field in fields.items() if field is None and c not in settings.values]
    if unset:
        names = ", ".join(f"airtable.values.{column}" for column in unset)
        raise ValueError(
            f"the AirTable table has no field for {', '.join(unset)}; set {names}"
        )
    return fields


class AirtableClient(Protocol):
    """Source of raw AirTable records (pyairtable ``RecordDict``) for one dataset."""

    def records(self, dataset: str) -> list[dict[str, Any]]: ...


class PyairtableClient:
    """Default client wrapping ``pyairtable``; reads the token from an env var."""

    def __init__(self, settings: AirtableSettings):
        try:
            from pyairtable import Api
            from pyairtable.formulas import match
        except ImportError as error:
            raise ImportError(
                "AirTable lookup needs the dca extra: pip install 'biahub[dca]'"
            ) from error
        token = os.environ.get(settings.token_env)
        if not token:
            raise RuntimeError(
                f"AirTable token not found: set the environment variable {settings.token_env}"
            )
        self._table = Api(token).table(settings.base_id, settings.table)
        self._match = match
        self._dataset_field = airtable_fields(settings)["dataset"]

    def records(self, dataset: str) -> list[dict[str, Any]]:
        return self._table.all(formula=self._match({self._dataset_field: dataset}))


def blank_row() -> dict[str, str]:
    """Return a row with every table column set to an empty string."""
    return {column: "" for column in TABLE_COLUMNS}


def _index_by_well(rows: list[dict[str, str]], origin: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        well = row["well"]
        if well in indexed:
            raise ValueError(f"Duplicate well {well} in {origin}")
        indexed[well] = row
    return indexed


def read_metadata_table(path: Path, dataset: str | None = None) -> dict[str, dict[str, str]]:
    """Read a CSV or TSV sample sheet into rows keyed by well (e.g. ``"C/2"``).

    With ``dataset`` set, keep only rows whose ``dataset`` cell equals it or is blank, so
    one sheet can serve several plates.
    """
    path = Path(path)
    sep = "\t" if path.suffix == ".tsv" else ","
    frame = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    missing = [column for column in TABLE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
    rows = frame[list(TABLE_COLUMNS)].to_dict(orient="records")
    for number, row in enumerate(rows, start=2):
        if not row["well"].strip():
            raise ValueError(f"{path} row {number} has a blank well")
    if dataset is not None:
        rows = [row for row in rows if row["dataset"].strip() in ("", dataset)]
    return _index_by_well(rows, str(path))


def _field_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return _field_to_str(value.get("name", ""))
    if isinstance(value, list):
        return ",".join(_field_to_str(item) for item in value)
    return str(value)


def _record_to_row(
    record: dict[str, Any], fields: dict[str, str | None], values: dict[str, str]
) -> dict[str, str]:
    cells = record.get("fields", {})
    row = blank_row()
    for column, field in fields.items():
        row[column] = values[column] if field is None else _field_to_str(cells.get(field))
    if not row["well"]:
        raise ValueError(f"AirTable record {record.get('id')} has no {fields['well']}")
    row["perturbation"] = AIRTABLE_PERTURBATION_ALIASES.get(
        row["perturbation"], row["perturbation"]
    )
    return row


def _collapse_fov_rows(well: str, rows: list[dict[str, str]]) -> dict[str, str]:
    differing = [c for c in TABLE_COLUMNS if len({row[c] for row in rows}) > 1]
    if differing:
        raise ValueError(
            f"Well {well} has FOV-level AirTable records that disagree on: "
            f"{', '.join(differing)}"
        )
    return rows[0]


def airtable_rows(
    client: AirtableClient, settings: AirtableSettings
) -> dict[str, dict[str, str]]:
    """Map the AirTable records of ``settings.dataset`` to rows keyed by well.

    A well-level record (blank or absent ``fov``) is the template and wins. A well with
    only FOV-level records is collapsed when all table columns agree, else ValueError.
    A ``perturbation`` of ``N/A`` (the lab table's uninfected wells) reads as ``uninfected``.
    """
    fields = airtable_fields(settings)
    dataset = settings.dataset
    records = client.records(dataset)
    if not records:
        raise ValueError(
            f"AirTable has no records with {fields['dataset']} = {dataset!r}; "
            "check airtable.dataset in the settings"
        )
    templates: dict[str, list[dict[str, str]]] = {}
    per_fov: dict[str, list[dict[str, str]]] = {}
    for record in records:
        row = _record_to_row(record, fields, settings.values)
        fov = _field_to_str(record.get("fields", {}).get(AIRTABLE_FOV_FIELD)).strip()
        target = per_fov if fov else templates
        target.setdefault(row["well"], []).append(row)

    rows: dict[str, dict[str, str]] = {}
    for well, candidates in templates.items():
        if len(candidates) > 1:
            raise ValueError(f"Duplicate well {well} in AirTable dataset {dataset}")
        rows[well] = candidates[0]
    for well, candidates in per_fov.items():
        if well not in rows:
            rows[well] = _collapse_fov_rows(well, candidates)
    return rows


def metadata_rows(
    plate: Path, settings: DcaFinalizeSettings, client: AirtableClient | None = None
) -> dict[str, dict[str, str]]:
    """Return one metadata row per well, from a sample sheet if present, else AirTable.

    A sheet keeps only rows for the plate's dataset stem (``<ds>`` for a projected
    ``<ds>_proj`` plate), and a sheet row with a blank ``dataset`` matches any plate.
    AirTable keeps the records of ``settings.airtable.dataset``.

    ``client`` overrides the default pyairtable client, which is built only when
    no sample sheet exists and ``settings.airtable`` is set.
    """
    plate = Path(plate)
    stem = dca_dataset_stem(plate)
    table = settings.metadata_table_path(plate)
    if table is not None:
        return read_metadata_table(table, dataset=stem)
    if settings.airtable is not None:
        client = client or PyairtableClient(settings.airtable)
        return airtable_rows(client, settings.airtable)
    expected = plate.parent / f"{stem}.dca-metadata.csv"
    raise FileNotFoundError(
        f"No metadata source: expected {expected} (or .tsv), or set airtable in the settings"
    )
