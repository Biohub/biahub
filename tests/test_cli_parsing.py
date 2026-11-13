import numpy as np
import typer

from iohub import open_ome_zarr
from typer.testing import CliRunner

from biahub.cli.parsing import MovingPositionDirpaths, ReferencePositionDirpaths

app = typer.Typer()


@app.command()
def _echo(
    moving_position_dirpaths: MovingPositionDirpaths,
    *,
    reference_position_dirpaths: ReferencePositionDirpaths,
):
    typer.echo(f"{len(moving_position_dirpaths)} {reference_position_dirpaths!r}")


def _position(tmp_path):
    with open_ome_zarr(
        tmp_path / "in.zarr", layout="hcs", mode="w", channel_names=["GFP"]
    ) as p:
        p.create_position("A", "1", "0")["0"] = np.zeros((1, 1, 2, 4, 4), dtype=np.float32)
    return str(tmp_path / "in.zarr" / "A" / "1" / "0")


def test_an_omitted_optional_reference_parses_to_an_empty_list(tmp_path):
    result = CliRunner().invoke(app, ["-m", _position(tmp_path)])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "1 []"


def test_a_path_matching_no_position_is_a_usage_error(tmp_path):
    result = CliRunner().invoke(app, ["-m", str(tmp_path / "missing.zarr" / "A" / "1" / "0")])
    assert result.exit_code == 2
    assert "no position directory found" in result.output
