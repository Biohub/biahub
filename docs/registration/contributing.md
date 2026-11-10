# Registration: changing the engine

How to add a method, a preprocessing block, a score metric or a setting to
`estimate-transform`, and the rules every change follows. The rules a test can check are
checked in `tests/test_registration_rules.py` (the skill's templates in
`tests/test_example_settings.py`); each rule below names its test.

## Words

- **moving** and **reference** are the two volumes: the moving volume is registered onto
  the reference. Code, settings and messages use these words (or `mov` / `ref` for short
  locals), never source/target or fixed. **source** means where something came from
  (a stand-in's timepoint, the file a transform was read from).
  *(test: `test_the_two_volumes_are_moving_and_reference`)*
- A **transform** is forward, moving -> reference, as `Transform` holds it; say an
  operation with `Transform`'s methods (`invert`, `@`, `on_grid`, `from_yx`, `covers`)
  rather than rebuilding it from matrices.
- A **method** estimates a transform; a **block** is one step of a stack or one
  preprocessing step; the **score** decides between candidates and flags timepoints.
- A setting's name is the name a config uses: a method's name is its settings section
  (`method: ants` reads `ants:`). *(test: `test_every_method_is_named_like_its_settings_section`)*

## Adding a method

A method is an `EstimatorBase` in `methods/<family>/<method>.py`, the family being
`landmark` (detect and match structures seen in both channels), `intensity` (compare the
images) or `manual` (a person or the setup decides).
*(test: `test_every_method_lives_in_its_familys_folder_under_its_own_name`)*

1. **Estimator.** Subclass `EstimatorBase`. `estimate(mov, ref, initial_transform)`
   applies its start and returns the full forward transform: a method that measures from
   the two images alone does it through `measure_from_start` (an identity start warps
   nothing); a landmark method moves its points by the start; a method that warps by
   itself (ANTs) passes it on. Only `shapes` computes a start instead (and comes first).
   Declare `holds_xy_across_gaps` and `accepts_competitor` when they apply. A method that
   scores its own choices (passes, candidates) gets a `tuning_metric` in its settings
   block (unset: the run's score). Raise `EstimationError` for a failed estimate (the run
   records it and moves on); anything else is a bug.
   *(test: `test_every_method_but_shapes_starts_from_its_initial_transform`)*
2. **Settings** (`biahub/settings.py`): a `<Method>Settings` block and its field on
   `EstimateTransformSettings`, the name in `EstimationMethod`, `METHOD_BLOCKS`,
   `METHOD_FAMILIES` (family, what it estimates) and `DEFAULT_SCORE_METRIC`; in
   `METHOD_TRANSFORM_TYPES` when it cannot fit every transform type; a default in
   `fill_method_block` unless the method has none to give (as `shapes`).
   *(tests: `test_every_method_has_a_family_and_says_what_it_estimates`,
   `test_every_method_has_a_default_score`)*
3. **Engine** (`engine/core.py`): its branch in `build_estimator`.
   *(test: `test_every_method_builds_from_its_default_settings`)*
4. **Resources.** Measure one job's peak memory (SLURM MaxRSS, or `/usr/bin/time -v` on a
   realistic pair) and its threads; add it to `RAM_VOLUMES_PER_JOB` / `CPUS_PER_JOB` when
   it differs from the defaults, with the measurement in the comment, ~30% headroom.
5. **The skill** (`.claude/skills/register-with-engine`): a row in the "Which method"
   table of `references/methods.md` and a `templates/estimate-transform-<method>.yml`
   that loads. *(tests: `test_the_skill_names_every_method`, `test_skill_config_templates_load`)*
6. **Tests**: the method recovers a known transform on a synthetic pair; it fails with
   `EstimationError` where it cannot work (empty frame, too few structures).

## Adding a preprocessing block, a score metric or a setting

- **A preprocessing block** names the volumes it applies to (`volumes: [moving,
  reference]`). An image block changes pixels in its volume's own grid; a mask block says
  where to compare. Give its reach (`blocks_footprint`) so the score keeps clear of the
  padding edge.
- **A score metric** goes in `INTENSITY_METRIC_FNS` or the bead metrics; it returns a
  float in a known range and nan when the volumes do not overlap.
- **A setting is read or refused.** A setting the run would ignore raises, saying why
  (`method X ignores ...`); a combination that cannot work raises at load, not mid-run.
  Use `RunError` for a run refused for its data; commands show it as a usage error.
- **Renaming.** A name from the legacy pipeline (on `main` before the engine) keeps a
  refusal naming the new one, and `convert-settings` rewrites it. A name only this engine
  introduced is renamed without one.

## Changes to how results are computed

- **Test first**: a test that fails before the change and passes after; for a
  refactor, a test that pins today's result before touching the code.
- **Same results**: a refactor or a speed-up shows identical outputs before and after,
  on the test suite and on runs of real data (records, transforms, evaluation). If the
  numbers change only by round-off, say so in the commit, with the size; do not keep a
  roundabout to avoid it. `tests/test_registration_golden.py` holds five synthetic runs'
  results (an initial matrix, shapes, propagation, repair, a file start); a change meant
  to move results rewrites them (`BIAHUB_UPDATE_GOLDEN=1 pytest
  tests/test_registration_golden.py`) and the file's diff is reviewed with the change.
- **Measure first**: an efficiency change starts from a profile on a realistic volume
  and ends with the same profile; a change that does not pay is dropped, and the
  measurement says why.

## Code

- Readable by a person and by an agent: say the operation directly, name things for
  what they are, a comment for the why. No detours that a reader has to undo.
- The library raises; it does not exit or prompt. Commands turn refusals into usage
  errors (`cli.parsing.refusals_as_usage_errors`).
- No personal paths or notes in code, docs, the skill or its templates.
