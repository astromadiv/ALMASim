/**
 * The simulation workflow: one `almasim simulation` run per chosen metadata
 * row, and what the estimate says the batch will take.
 */
import { buildArgv, initialValues, type Command, type Values } from './commandForm';
import { RUN_DIR } from './runDir';

/** The simulation form's fields the page sets from the chosen rows. */
export const ROW_FIELDS = ['metadata_path', 'row_idx'];

/**
 * The console's starting values over the CLI's. The CLI writes an ML shard
 * to one file in your home and keeps nothing else; the console runs one job
 * per row at once, so each writes its shard, and keeps its FITS products,
 * in its own run folder. It also computes the channels in
 * parallel, with the job's CPUs (`simulationCpus`).
 */
export const CONSOLE_DEFAULTS: Values = {
	output_dir: `${RUN_DIR}/outputs`,
	ml_shard_path: `${RUN_DIR}/ml_shard.h5`,
	save_mode: 'fits',
	persist_standard_outputs: true,
	// Channels in parallel, one worker per CPU of the job (simulationCpus).
	backend: 'local'
};

/** The form's starting values: the CLI's, then the console's where the command has the field. */
export function simulationDefaults(command: Command): Values {
	const values = initialValues(command);
	for (const [name, value] of Object.entries(CONSOLE_DEFAULTS)) {
		if (name in values) values[name] = value;
	}
	return values;
}

/** An input cube to simulate, as `/api/v2/simulation/cubes` lists it. */
export interface CubeRun {
	path: string;
	label: string;
}

/**
 * One run's command line: the form, for one row of the metadata CSV, and the
 * input cube it simulates, if any (which sets its own size: the size options
 * are left out). In a batch, a shared output folder gets a folder per run,
 * named after its row or its cube, so runs that go at once never write over
 * each other.
 */
export function simulationArgv(
	command: Command,
	values: Values,
	metadataPath: string,
	rowIdx: number,
	batch = false,
	cube: CubeRun | null = null
): string[] {
	const run: Values = { ...values, metadata_path: metadataPath, row_idx: String(rowIdx) };
	if (cube) Object.assign(run, { input_path: cube.path, n_pix: '', n_channels: '' });
	const out = typeof values.output_dir === 'string' ? values.output_dir : '';
	if (batch && out && !out.includes(RUN_DIR))
		run.output_dir = `${out.replace(/\/+$/, '')}/${cube ? cube.label : `row-${rowIdx}`}`;
	return buildArgv(command, run);
}

/** The estimate for the batch: per simulation, times runs per row, times rows. */
export function batchGb(perSimulationGb: number, rows: number, perRow: number): number {
	return perSimulationGb * Math.max(rows, 0) * Math.max(perRow, 1);
}

/**
 * The CPUs a run's job asks for: one per worker with the `local` backend,
 * which images channels in parallel processes on the job's node, and one
 * with `sync`, which images them one after another. So the job asks Slurm
 * for exactly what the run uses.
 */
export function simulationCpus(values: Values): number {
	if (values.backend !== 'local') return 1;
	return Math.max(1, Math.floor(Number(values.n_workers)) || 1);
}

/** A cube's volume as a run makes it: square, the larger side, times its channels. */
export function cubeVolume(cube: { n_pix_x: number; n_pix_y: number; n_channels?: number | null }) {
	return Math.max(cube.n_pix_x, cube.n_pix_y) ** 2 * (cube.n_channels || 1);
}

/**
 * The batch's total when its input cubes differ in size: the first run's
 * estimate, scaled by each cube's volume, times the runs per cube.
 */
export function cubesGb(perFirstGb: number, volumes: number[], perRow: number): number {
	if (!volumes.length || !volumes[0]) return 0;
	const total = volumes.reduce((sum, v) => sum + v, 0);
	return (perFirstGb * total * Math.max(perRow, 1)) / volumes[0];
}
