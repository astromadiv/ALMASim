import { describe, expect, it } from 'vitest';
import {
	candidateValues,
	chainFolder,
	chainSteps,
	defaultFolder,
	lastImaging,
	lastSearch,
	newestValidation,
	resolvedProducts,
	sig,
	verdict,
	windowsArgv
} from './noise';

const run = (argv: string[], state = 'succeeded', run_dir = '/runs/R') => ({
	argv,
	state,
	run_dir
});

describe('the runs the page starts from', () => {
	const runs = [
		run(['image', 'image-from-ms', '/p2.csv', '/out2'], 'failed'),
		run(['image', 'image-from-ms', '/p1.csv', '/out1', '--cube']),
		run(['metadata', '--band', '3', '--save-csv', '{run_dir}/results.csv'], 'succeeded', '/runs/M'),
		run(
			['products', 'resolve', '--save-products-csv', '{run_dir}/products.csv'],
			'succeeded',
			'/runs/A'
		),
		run(
			['products', 'resolve', '--save-products-csv', '{run_dir}/products.csv'],
			'failed',
			'/runs/B'
		),
		run(
			['products', 'resolve', '--save-products-csv', '{run_dir}/products.csv'],
			'succeeded',
			'/runs/A'
		)
	];

	it('are the newest that succeeded', () => {
		expect(lastImaging(runs)).toEqual({ parameters: '/p1.csv', images: '/out1' });
		expect(lastSearch(runs)).toBe('/runs/M/results.csv');
		expect(resolvedProducts(runs)).toEqual(['/runs/A/products.csv']);
		expect(lastImaging([])).toBeNull();
		expect(lastSearch([])).toBeNull();
	});
});

describe('the chain', () => {
	const source = {
		parameters: '/p.csv',
		images: '/data/x-images',
		metadata: '/m.csv',
		products: ['/a.csv', '/b.csv']
	};

	it('lists the windows into its run folder first', () => {
		expect(windowsArgv(source)).toEqual([
			'noise', 'windows', '/p.csv', '--images', '/data/x-images', '--metadata', '/m.csv',
			'--products', '/a.csv', '--products', '/b.csv', '--out', '{run_dir}'
		]); // prettier-ignore
	});

	it('mirrors noise_model_from_windows, with the chosen windows and partial steps', () => {
		const steps = chainSteps(source, ['w1', 'w2'], '/data/x-noise/', ' ');
		expect(steps.map((s) => s.key)).toEqual(['windows', 'extract', 'fit', 'aggregate', 'validate']);
		expect(steps[0].argv.slice(-4)).toEqual(['--out', '/data/x-noise', '--names', 'w1,w2']);
		expect(steps[1].argv).toEqual([
			'noise', 'extract', '--windows', '/data/x-noise/windows.csv', '--out', '/data/x-noise',
			'--allow-partial'
		]); // prettier-ignore
		expect(steps[3].argv).toEqual([
			'noise', 'aggregate', '--out', '/data/x-noise', '--from', 'alma', '--output',
			'/data/x-noise/candidate.json'
		]); // prettier-ignore
		expect(steps[4].argv).toContain('/data/x-noise/rows.csv');
		expect(chainFolder(steps[0].argv)).toBe('/data/x-noise');
		expect(defaultFolder('/data/x-images/')).toBe('/data/x-noise');
		expect(defaultFolder('/data/cubes')).toBe('/data/cubes-noise');
		expect(defaultFolder(' ')).toBe('');
	});
});

describe('what the chain made', () => {
	it('reads the candidate', () => {
		const model = {
			provenance: {
				notes: "a: 1 window(s), kept the base's; b: 0 window(s), kept the base's",
				values: { 'x.rho1': { n: 3, median: 0.6667, q16: 0.66, q84: 0.68 } }
			}
		};
		expect(candidateValues(model)).toEqual({
			values: [{ key: 'x.rho1', n: 3, median: 0.6667, q16: 0.66, q84: 0.68 }],
			kept: ["a: 1 window(s), kept the base's", "b: 0 window(s), kept the base's"]
		});
		expect(candidateValues(null)).toEqual({ values: [], kept: [] });
	});

	it('reads the newest validation', () => {
		const comparison = {
			passed: false,
			windows: 4,
			model: { label: 'alma-candidate@2' },
			metrics: {
				'cont.sigma': {
					kind: 'ratio',
					n: 4,
					median: 1.2,
					q16: 1.1,
					q84: 1.3,
					p: 0.01,
					within_10pct: 1
				}
			},
			checks: [{ name: 'continuum noise', passed: false, detail: '1.2' }]
		};
		const v = verdict(comparison);
		expect(v?.passed).toBe(false);
		expect(v?.metrics[0]).toMatchObject({ metric: 'cont.sigma', within10: 1, p: 0.01 });
		expect(v?.checks).toEqual([{ name: 'continuum noise', passed: false, detail: '1.2' }]);
		expect(verdict({})).toBeNull();
		expect(
			newestValidation([
				{ name: 'aaa', kind: 'folder', modified: 1 },
				{ name: 'bbb', kind: 'folder', modified: 3 },
				{ name: 'model.json', kind: 'file', modified: 9 }
			])
		).toBe('bbb');
		expect(newestValidation([])).toBeNull();
		expect(sig(0.123456)).toBe('0.123');
		expect(sig(null)).toBe('—');
	});
});
