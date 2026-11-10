// Reconstruct subworkflows: init, transfer functions, and fan-out × N positions
// × M reconstructions.
//
// This module is PATH-AGNOSTIC. Callers pass the input zarr, output zarr, and
// config explicitly; the module has no idea where it sits in the pipeline
// directory layout. The transfer functions live in `transfer_functions/` beside
// the output zarr, one `<name>.zarr` per reconstruction (the module's internal
// convention).
//
// The config lists named waveorder reconstructions under `reconstructions:`
// (biahub.settings.ReconstructSettings), e.g. `phase` from BF plus `gfp` and
// `mcherry` deconvolutions. Each one is its own transfer-function task and its
// own per-position tasks, and all of them write into the one output plate. The
// config is the single source of truth and must carry the pixel sizes; the CLI
// validates it and only warns if those pixel sizes disagree with the input zarr
// metadata. No resolved config is written.
//
// Three phases, THREE SUBWORKFLOWS, because the pipeline schedules each of them
// at a different time (see mantis-v2.nf):
// 1. reconstruct_init_wf: validates the config, creates the output plate with
//    every reconstruction's channels, emits one RESOURCES line per
//    reconstruction. Metadata-only, head node, part of the up-front init phase.
// 2. reconstruct_tf_wf: one transfer-function task per reconstruction. This is
//    real compute, but it depends only on the input plate's SHAPE — waveorder
//    builds the transfer function from the first position's dimensions, reading
//    no pixels — so the pipeline starts it as soon as the input store has been
//    SCAFFOLDED, and it overlaps the upstream steps' compute instead of sitting
//    on the critical path between them and reconstruction.
// 3. reconstruct_run_wf: per-position inverse TF application of the
//    reconstructions it is given, each gated on its own transfer function. A
//    reconstruction whose waveorder `device` is a GPU runs on one.
//
// run_apply_inv_tf uses `--cluster debug` so that submitit's DebugExecutor runs
// the work in-process.  Nextflow already handles per-position fan-out and
// resource scheduling, so the CLI must NOT submit its own SLURM jobs.
// See: examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md

include { slurm_logs; slurm_log_dir; slurm_output_readme; retry_time; retry_memory } from './common'


process init_apply_inv_tf {
    label 'cpu_local'

    input:
    val input_zarr
    val output_zarr
    val config
    path config_file  // staged only for the task hash: see common.nf, #397
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('reconstruct')}"
    ${slurm_output_readme('reconstruct', output_zarr)}
    biahub apply-inv-tf --init \
        -i "${input_zarr}"/*/*/* \
        -o "${output_zarr}" \
        -c "${config}"
    """
}

process compute_transfer_function {
    tag "${meta.reconstruction}"
    label 'cpu'
    clusterOptions { slurm_logs('reconstruct') }
    // The memory comes from `biahub apply-inv-tf --init` (tf_mem_gb), per
    // reconstruction:
    //
    // - Phase, birefringence and widefield fluorescence get 64 GB. waveorder
    //   upsamples the volume to Nyquist internally before building the TF
    //   (waveorder/models/phase_thick_3d.py::calculate_transfer_function); the
    //   peak footprint scales with that upsampled volume, not the input volume.
    //   For properly-sampled label-free data the Nyquist upsampling factor is 1
    //   in every axis, so even for the largest volume we expect
    //   (~2048 x 2048 x 128 = 2 GB float32) the phase TF needs ~64 GB (2 GB x
    //   waveorder's x32 Fourier multiplier).
    // - A confocal-pinhole fluorescence TF scales with the z-padded volume,
    //   ~1 kB per voxel: 113 GB measured for a mantis V2 neuromast stack.
    //
    // The TF computation is torch-CPU-FFT-bound (large 3D FFTs over the
    // upsampled volume in optics.compute_weak_object_transfer_function_3D) and
    // is not thread-pinned in the compute-tf path, so torch parallelizes the
    // FFTs across the granted cores.  8 sits in the sweet spot before FFT
    // thread-scaling tails off. waveorder builds every transfer function on the
    // CPU, whatever the reconstruction's `device`.
    cpus 8
    memory { retry_memory(meta.tf_mem_gb, task) }
    time { retry_time(30, task) }

    input:
    val meta
    val trigger
    val input_zarr
    val tf_dir
    val config
    path config_file  // staged only for the task hash: see common.nf, #397

    output:
    val meta

    script:
    """
    biahub compute-tf \
        -i "${input_zarr}"/*/*/* \
        -o "${tf_dir}" \
        -c "${config}" \
        -r "${meta.reconstruction}"
    """
}

// --resume: a preempted task finishes the timepoint it is writing and stops
// early, so the retry (or a later `nextflow -resume`) recomputes only the
// timepoints this position had not finished. waveorder keys each finished
// timepoint on the reconstruction settings and on the transfer function, so a
// config change or a recomputed transfer function recomputes instead of being
// skipped.
def apply_inv_tf_script(position, meta, input_zarr, output_zarr, tf_dir, config) {
    """
    biahub apply-inv-tf --cluster debug --resume \
        -i "${input_zarr}/${position}" \
        -t "${tf_dir}" \
        -o "${output_zarr}" \
        -c "${config}" \
        -r "${meta.reconstruction}"
    """
}

process run_apply_inv_tf {
    tag "${position} ${meta.reconstruction}"
    label 'cpu'
    clusterOptions { slurm_logs('reconstruct') }
    cpus { meta.cpus }
    memory { retry_memory(meta.mem_gb, task) }
    time { retry_time(meta.time_minutes, task) }

    input:
    tuple val(position), val(meta)
    val input_zarr
    val output_zarr
    val tf_dir
    val config
    path config_file  // staged only for the task hash: see common.nf, #397

    output:
    tuple val(position), val(meta.reconstruction)

    script:
    apply_inv_tf_script(position, meta, input_zarr, output_zarr, tf_dir, config)
}

// The same task on a GPU, for a reconstruction with a GPU `device` (e.g. the
// Richardson-Lucy deconvolutions). A separate process because the label, which
// routes the task in the slurm profile, cannot depend on the task's input.
process run_apply_inv_tf_gpu {
    tag "${position} ${meta.reconstruction}"
    // Preemptable like run_apply_inv_tf: --resume makes a reclaimed task cost at
    // most the timepoint it was reconstructing. Any GPU node of `preempted`; one
    // GPU per task, because waveorder runs a GPU reconstruction one timepoint at
    // a time (the CLI sizes it to one worker).
    label 'gpu_preempted'
    clusterOptions { "--gres=gpu:1 " + slurm_logs('reconstruct') }
    cpus { meta.cpus }
    memory { retry_memory(meta.mem_gb, task) }
    time { retry_time(meta.time_minutes, task) }

    input:
    tuple val(position), val(meta)
    val input_zarr
    val output_zarr
    val tf_dir
    val config
    path config_file  // staged only for the task hash: see common.nf, #397

    output:
    tuple val(position), val(meta.reconstruction)

    script:
    apply_inv_tf_script(position, meta, input_zarr, output_zarr, tf_dir, config)
}


// The transfer-function directory, derived from the output plate's parent
// directory. Exposed as a function because both reconstruct_tf_wf and
// reconstruct_run_wf need it and the pipeline invokes them separately.
def transfer_functions_path(output_zarr) {
    return "${new File(output_zarr).parent}/transfer_functions"
}

// Every RESOURCES line of `apply-inv-tf --init`, one per reconstruction, in
// config order: [reconstruction, channels, gpus, cpus, mem_gb, time_minutes,
// tf_mem_gb]. parse_resources (common.nf) reads only the last line, which is one
// reconstruction's.
def parse_reconstruction_resources(stdout_text) {
    def prefix = 'RESOURCES:'
    def lines = stdout_text.trim().readLines().findAll { line -> line.startsWith(prefix) }
    if (!lines) {
        error "Expected '${prefix}' lines in `biahub apply-inv-tf --init` output but none was found. The CLI may have failed."
    }
    def slurper = new groovy.json.JsonSlurper()
    return lines.collect { line ->
        def res = slurper.parseText(line.substring(prefix.length()).trim())
        [
            reconstruction: res.reconstruction as String,
            channels:       res.channels.collect { c -> c as String },
            gpus:           res.gpus as int,
            cpus:           res.cpus as int,
            mem_gb:         res.mem_gb as int,
            time_minutes:   res.time_minutes as int,
            tf_mem_gb:      res.tf_mem_gb as int,
        ]
    }
}


// Validate the config and scaffold the output plate. Metadata-only and cheap,
// so it belongs in the pipeline's up-front init phase.
//
// take:
//   input_zarr   path to the input plate.zarr (deskew output)
//   output_zarr  path to the reconstructed output plate.zarr
//   config       path to the reconstruct settings YAML
//   trigger      gating channel — init starts once this emits
// emit:
//   resources    ONE list of per-reconstruction RESOURCES maps (see
//                parse_reconstruction_resources), in config order
//   done         fires once the output plate exists
workflow reconstruct_init_wf {
    take:
    input_zarr
    output_zarr
    config
    trigger

    main:
    init_out = init_apply_inv_tf(input_zarr, output_zarr, config, file(config), trigger.collect().map { 'done' })

    emit:
    resources = init_out.map { stdout_text -> parse_reconstruction_resources(stdout_text) }
    done      = init_out.map { 'done' }
}


// Compute every reconstruction's transfer function, one task each.
//
// take:
//   input_zarr   path to the input plate.zarr (deskew output); only its shape
//                is read, so a scaffolded-but-empty store is enough
//   output_zarr  path to the reconstructed output plate.zarr (locates the TFs)
//   config       path to the reconstruct settings YAML
//   resources    the per-reconstruction list from reconstruct_init_wf
//   trigger      gating channel — the TFs start once this emits
// emit:
//   done         one RESOURCES map per reconstruction, emitted when its
//                transfer function exists
workflow reconstruct_tf_wf {
    take:
    input_zarr
    output_zarr
    config
    resources
    trigger

    main:
    tf_dir = transfer_functions_path(output_zarr)
    reconstructions = resources.flatMap { entries -> entries }
    tf_done = compute_transfer_function(reconstructions, trigger.collect().map { 'done' },
                                        input_zarr, tf_dir, config, file(config))

    emit:
    done = tf_done
}


// Fan out one inverse-TF task per position for each reconstruction it is given.
//
// take:
//   positions    collected channel of position keys (e.g. ['A/1/0', 'B/1/0'])
//   input_zarr   path to the input plate.zarr (deskew output)
//   output_zarr  path to the reconstructed output plate.zarr
//   config       path to the reconstruct settings YAML
//   tf_done      the reconstructions to run: a RESOURCES map per reconstruction,
//                each emitted once its transfer function exists (a subset of
//                reconstruct_tf_wf's `done`)
//   prev_done    gating channel — the input store holds data
// emit:
//   done         fires once every task has finished; also when it was given no
//                reconstruction, so a caller can always gate on it
workflow reconstruct_run_wf {
    take:
    positions
    input_zarr
    output_zarr
    config
    tf_done
    prev_done

    main:
    tf_dir = transfer_functions_path(output_zarr)

    // GATE CHANNELS ARE MAPPED TO A TOKEN BEFORE COMBINING, never combined raw.
    // `combine` FLATTENS a list-valued item into the tuple, so what the producer
    // happens to emit leaks into the tuple's arity: a step's `done` is a COLLECTED
    // list of every position, which turned [pos, meta] into
    // [pos, meta, p1, p2, … p54] and blew up a three-parameter closure with
    // `MissingMethodException` — after the previous step had already run. Mapping
    // reads nothing out of the gate, so no producer's payload shape can reach here.
    // The RESOURCES map is not a list, so it combines as one element.
    tasks = tf_done
        .combine(positions.flatMap { items -> items })
        .combine(prev_done.map { 'done' })
        .map { meta, pos, _gate -> [pos, meta] }
        .branch { _pos, meta ->
            gpu: meta.gpus > 0
            cpu: true
        }

    cpu_done = run_apply_inv_tf(tasks.cpu, input_zarr, output_zarr, tf_dir, config, file(config))
    gpu_done = run_apply_inv_tf_gpu(tasks.gpu, input_zarr, output_zarr, tf_dir, config, file(config))

    // `collect` emits nothing for an empty channel, so a caller handed no
    // reconstruction would wait forever on a bare collect.
    rc_done = cpu_done.mix(gpu_done).collect().ifEmpty([])

    emit:
    done = rc_done
}
