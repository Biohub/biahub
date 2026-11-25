#!/usr/bin/env bash
# Apply a transforms file (from estimate-transform) to positions.
#
# Copy this next to the transforms file and fill in the variables below. Keep it there:
# it is the record of how the output was produced.
#
# With REFERENCE set: registration -- every moving position is transformed onto the
# reference grid; the reference channels are copied and every moving channel transformed,
# except one the reference store also has and the file does not list in moving_channels
# (set CHANNELS to choose). Without REFERENCE: stabilization -- each store is
# transformed onto its own grid. Timepoints written with a non-accepted transform are
# printed and recorded in the output metadata.
set -euo pipefail

BIAHUB=/path/to/biahub                         # checkout on main
MOVING=("/path/to/deskew.zarr/*/*/*")          # positions to transform: one or more quoted globs
REFERENCE=("/path/to/reconstruct.zarr/*/*/*")  # the same for the reference; () to stabilize
TRANSFORMS=./transforms.yml
OUTPUT=/path/to/registered.zarr
CHANNELS=()                                    # e.g. ("GFP EX488 EM525-45"); empty = all
                                               # -- set it when the moving store mixes cameras
                                               #    (DragonFly: the BF is on the reference camera)
TIMEPOINTS=""                                  # e.g. "0,40,80,120"; empty = every timepoint. The
                                               # output holds only these, in order (indices recorded
                                               # in its biahub-apply-transform metadata)
CROP_TO_OVERLAP=false                          # true: crop to the box every transform covers
SBATCH=""                                      # optional sbatch file (time, partition -- default
                                               # preempted); --mem-per-cpu is per CPU, x up to 32
VENV=""                           # empty: $BIAHUB/.venv; or another biahub venv (runs BIAHUB's code)

# Relative paths are next to this script, wherever it is launched from.
cd "$(dirname "$0")"
source "${VENV:-$BIAHUB/.venv}/bin/activate"
export PYTHONPATH="$BIAHUB${PYTHONPATH:+:$PYTHONPATH}"   # run this checkout, whatever the venv

# Expand each glob into one argument per position, keeping spaces inside a path.
expand() { for glob in "$@"; do compgen -G "$glob"; done; }
mapfile -t moving < <(expand "${MOVING[@]}")
[[ ${#moving[@]} -gt 0 ]] || { echo "no positions match MOVING=${MOVING[*]}" >&2; exit 1; }
args=(-m "${moving[@]}" -c "$TRANSFORMS" -o "$OUTPUT" --cluster slurm --monitor)
if [[ ${#REFERENCE[@]} -gt 0 ]]; then
    mapfile -t reference < <(expand "${REFERENCE[@]}")
    [[ ${#reference[@]} -gt 0 ]] || { echo "no positions match REFERENCE=${REFERENCE[*]}" >&2; exit 1; }
    args+=(-r "${reference[@]}")
fi
[[ -n "$TIMEPOINTS" ]] && args+=(--timepoints "$TIMEPOINTS")
for channel in "${CHANNELS[@]}"; do args+=(--channels "$channel"); done
[[ "$CROP_TO_OVERLAP" == true ]] && args+=(--crop-to-overlap)
[[ -n "$SBATCH" ]] && args+=(-sb "$SBATCH")

biahub apply-transform "${args[@]}" "$@"
