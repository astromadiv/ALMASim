import { describe, expect, it } from 'vitest';

import type { Command, Param } from './commandForm';
import {
	batchGb,
	cubeVolume,
	cubesGb,
	simulationArgv,
	simulationCpus,
	simulationDefaults
} from './simulate';

const option = (name: string, extra: Partial<Param> = {}): Param => ({
	name,
	kind: 'option',
	flags: [`--${name.replace(/_/g, '-')}`],
	negative_flags: [],
	required: false,
	multiple: false,
	default: null,
	help: '',
	panel: null,
	type: 'string',
	...extra
});

const SIMULATION: Command = {
	name: 'simulation',
	summary: '',
	help: '',
	executor: 'slurm',
	needs_internet: false,
	extra_args: false,
	params: [
		option('metadata_path', { required: true, type: 'path' }),
		option('row_idx', { type: 'integer' }),
		option('source_type', { type: 'choice', choices: ['point', 'gaussian'], default: 'point' }),
		option('n_pix', { type: 'integer' })
	]
};

describe('simulationArgv', () => {
	it('runs the form on one row of the metadata', () => {
		const values = { metadata_path: '', row_idx: '', source_type: 'gaussian', n_pix: '128' };
		expect(simulationArgv(SIMULATION, values, '/data/meta.csv', 3)).toEqual([
			'simulation',
			'--metadata-path', '/data/meta.csv',
			'--row-idx', '3',
			'--source-type', 'gaussian',
			'--n-pix', '128'
		]); // prettier-ignore
	});
});

describe('batchGb', () => {
	it('multiplies by rows and runs per row', () => {
		expect(batchGb(0.5, 4, 2)).toBe(4);
		expect(batchGb(0.5, 4, 0)).toBe(2);
		expect(batchGb(0.5, 0, 3)).toBe(0);
	});
});

describe('the console defaults', () => {
	const command: Command = {
		...SIMULATION,
		params: [
			...SIMULATION.params,
			option('output_dir', { default: '/home/me/almasim_outputs/simulation', type: 'path' }),
			option('ml_shard_path', { default: '/home/me/almasim_outputs/ddrm.h5', type: 'path' }),
			option('save_mode', { type: 'choice', choices: ['memory', 'fits'], default: 'memory' }),
			option('persist_standard_outputs', { type: 'boolean', default: false })
		]
	};

	it('keep each run in its own folder, its FITS products kept', () => {
		const values = simulationDefaults(command);
		expect(values).toMatchObject({
			output_dir: '{run_dir}/outputs',
			ml_shard_path: '{run_dir}/ml_shard.h5',
			save_mode: 'fits',
			persist_standard_outputs: true
		});
		expect(simulationArgv(command, values, '/m.csv', 0)).toEqual([
			'simulation', '--metadata-path', '/m.csv', '--row-idx', '0',
			'--output-dir', '{run_dir}/outputs',
			'--ml-shard-path', '{run_dir}/ml_shard.h5',
			'--save-mode', 'fits',
			'--persist-standard-outputs'
		]); // prettier-ignore
	});

	it('give each row its own folder when a batch shares one', () => {
		const values = { ...simulationDefaults(command), output_dir: '/data/sims/' };
		const argv = simulationArgv(command, values, '/m.csv', 4, true);
		expect(argv[argv.indexOf('--output-dir') + 1]).toBe('/data/sims/row-4');
		const alone = simulationArgv(command, values, '/m.csv', 4, false);
		expect(alone[alone.indexOf('--output-dir') + 1]).toBe('/data/sims/');
	});
});

describe('simulationCpus', () => {
	it('asks for one CPU per local worker', () => {
		expect(simulationCpus({ backend: 'local', n_workers: '16' })).toBe(16);
		expect(simulationCpus({ backend: 'local', n_workers: '' })).toBe(1);
		expect(simulationCpus({ backend: 'local', n_workers: '2.7' })).toBe(2);
	});

	it('asks for one CPU when channels go one after another', () => {
		expect(simulationCpus({ backend: 'sync', n_workers: '16' })).toBe(1);
		expect(simulationCpus({})).toBe(1);
	});

	it('is local by default in the console', () => {
		const command: Command = {
			...SIMULATION,
			params: [
				...SIMULATION.params,
				option('backend', { type: 'choice', choices: ['sync', 'local'], default: 'sync' }),
				option('n_workers', { type: 'integer', default: 1 })
			]
		};
		const values = simulationDefaults(command);
		expect(values.backend).toBe('local');
		expect(simulationCpus(values)).toBe(1); // until the page sets the site's CPUs
	});
});

describe('runs from input cubes', () => {
	const command: Command = {
		...SIMULATION,
		params: [
			...SIMULATION.params,
			option('input_path', { type: 'path', flags: ['--input'] }),
			option('n_channels', { type: 'integer' }),
			option('output_dir', { type: 'path' })
		]
	};
	const cube = { path: '/data/cubes/x.ms.split/SPW-1/wsclean-cube-model.fits', label: 'x_SPW-1' };

	it('give the cube and leave its size to it', () => {
		const values = { ...simulationDefaults(command), n_pix: '64', n_channels: '8' };
		const argv = simulationArgv(command, values, '/m.csv', 3, false, cube);
		expect(argv).toContain('--input');
		expect(argv[argv.indexOf('--input') + 1]).toBe(cube.path);
		expect(argv).not.toContain('--n-pix');
		expect(argv).not.toContain('--n-channels');
	});

	it('get a folder each, named after the cube, in a shared output folder', () => {
		const values = { ...simulationDefaults(command), output_dir: '/data/sims' };
		const argv = simulationArgv(command, values, '/m.csv', 3, true, cube);
		expect(argv[argv.indexOf('--output-dir') + 1]).toBe('/data/sims/x_SPW-1');
	});

	it('add up by each cube s volume', () => {
		expect(cubeVolume({ n_pix_x: 10, n_pix_y: 20, n_channels: 3 })).toBe(1200);
		expect(cubesGb(2, [100, 300], 1)).toBe(8); // the first is 2 GB, the second three times it
		expect(cubesGb(2, [100, 300], 2)).toBe(16);
		expect(cubesGb(2, [], 1)).toBe(0);
	});
});
