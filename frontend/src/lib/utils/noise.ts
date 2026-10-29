/**
 * The noise-model workflow: an imaging run's windows (`noise windows`), then
 * extract → fit → aggregate → validate as a chain of jobs, each after the
 * one before (almasim.jobs.pipeline.noise_model_from_windows), and what the
 * page reads back: the candidate model and the validation's comparison.
 */
import { inRunDir, RUN_DIR } from './runDir';

export const WINDOWS_CSV = 'windows.csv';
export const ROWS_CSV = 'rows.csv';
export const CANDIDATE_JSON = 'candidate.json';
export const COMPARISON_JSON = 'comparison.json';
/** Each step's failed windows (almasim.services.noise.batch.run_tasks). */
export const FAILURES: Record<string, string> = {
	extract: 'failures.json',
	fit: 'failures-fit.json',
	validate: 'failures-validate.json'
};

/** The runs the page reads; a subset of the API's Run. */
export interface RunLike {
	argv: string[];
	state: string;
	run_dir: string;
}

export interface WindowsSource {
	/** The imaging run's parameters CSV and output folder. */
	parameters: string;
	images: string;
	/** The archive search's results CSV and resolved-products CSVs. */
	metadata: string;
	products: string[];
}

/** The value `argv` gives `flag`, or null. */
export function optionValue(argv: readonly string[], flag: string): string | null {
	const i = argv.indexOf(flag);
	return i >= 0 ? (argv[i + 1] ?? null) : null;
}

const succeeded = (runs: readonly RunLike[], command: string[]) =>
	runs.filter((r) => r.state === 'succeeded' && command.every((word, i) => r.argv[i] === word));

/** The newest succeeded imaging run's parameters CSV and output folder (runs newest first). */
export function lastImaging(
	runs: readonly RunLike[]
): { parameters: string; images: string } | null {
	const run = succeeded(runs, ['image', 'image-from-ms'])[0];
	return run && run.argv[2] && run.argv[3]
		? { parameters: run.argv[2], images: run.argv[3] }
		: null;
}

/** The newest succeeded archive search's results CSV. */
export function lastSearch(runs: readonly RunLike[]): string | null {
	for (const run of succeeded(runs, ['metadata'])) {
		const csv = optionValue(run.argv, '--save-csv');
		if (csv) return inRunDir(csv, run.run_dir);
	}
	return null;
}

/** Every succeeded resolve's products CSV: they link each MS to its observation. */
export function resolvedProducts(runs: readonly RunLike[]): string[] {
	const paths = succeeded(runs, ['products', 'resolve']).flatMap((run) => {
		const csv = optionValue(run.argv, '--save-products-csv');
		return csv ? [inRunDir(csv, run.run_dir)] : [];
	});
	return [...new Set(paths)];
}

/** `noise windows` over an imaging run, into `out` (default: the run's own folder). */
export function windowsArgv(source: WindowsSource, out = RUN_DIR, names?: string[]): string[] {
	const argv = [
		'noise',
		'windows',
		source.parameters,
		'--images',
		source.images,
		'--metadata',
		source.metadata,
		...source.products.flatMap((csv) => ['--products', csv]),
		'--out',
		out
	];
	if (names?.length) argv.push('--names', names.join(','));
	return argv;
}

export interface Step {
	key: 'windows' | 'extract' | 'fit' | 'aggregate' | 'validate';
	title: string;
	argv: string[];
}

/**
 * The chain into `folder`: the chosen windows' table, then the four steps of
 * almasim.jobs.pipeline.noise_model_from_windows. A window failing does not
 * stop the chain (--allow-partial); a step fails only when all its windows do.
 */
export function chainSteps(
	source: WindowsSource,
	names: string[],
	folder: string,
	base: string
): Step[] {
	const f = folder.replace(/\/+$/, '');
	const windows = `${f}/${WINDOWS_CSV}`;
	const candidate = `${f}/${CANDIDATE_JSON}`;
	return [
		{ key: 'windows', title: 'Table the windows', argv: windowsArgv(source, f, names) },
		{
			key: 'extract',
			title: "Measure the windows' noise",
			argv: ['noise', 'extract', '--windows', windows, '--out', f, '--allow-partial']
		},
		{
			key: 'fit',
			title: 'Fit their calibration residuals',
			argv: ['noise', 'fit', '--windows', windows, '--out', f, '--allow-partial']
		},
		{
			key: 'aggregate',
			title: 'Reduce into a candidate model',
			argv: [
				'noise',
				'aggregate',
				'--out',
				f,
				'--from',
				base.trim() || 'alma',
				'--output',
				candidate
			]
		},
		{
			key: 'validate',
			title: 'Validate the candidate',
			argv: [
				'noise',
				'validate',
				'--model',
				candidate,
				'--windows',
				windows,
				'--metadata',
				`${f}/${ROWS_CSV}`,
				'--measured',
				f,
				'--out',
				f,
				'--allow-partial'
			]
		}
	];
}

/** The folder the chain writes to, from its first step's command line. */
export function chainFolder(windowsRunArgv: readonly string[]): string | null {
	return optionValue(windowsRunArgv, '--out');
}

/** A default folder for the chain: beside the imaging output. */
export function defaultFolder(images: string): string {
	const dir = images.trim().replace(/\/+$/, '');
	return dir ? `${dir.replace(/-images$/, '')}-noise` : '';
}

export interface ValueRow {
	key: string;
	median: number;
	q16: number;
	q84: number;
	n: number;
}

/** The values a candidate took from the windows (its provenance), and what it kept. */
export function candidateValues(model: unknown): { values: ValueRow[]; kept: string[] } {
	const provenance = (model as { provenance?: Record<string, unknown> } | null)?.provenance;
	const values = (provenance?.values ?? {}) as Record<string, Omit<ValueRow, 'key'>>;
	const notes = typeof provenance?.notes === 'string' ? provenance.notes : '';
	return {
		values: Object.entries(values).map(([key, v]) => ({ key, ...v })),
		kept: notes
			.split(';')
			.map((n) => n.trim())
			.filter(Boolean)
	};
}

export interface MetricRow {
	metric: string;
	kind: 'ratio' | 'difference';
	n: number;
	median: number;
	q16: number;
	q84: number;
	p: number | null;
	within10: number | null;
}

export interface Verdict {
	passed: boolean;
	windows: number;
	label: string;
	metrics: MetricRow[];
	checks: { name: string; passed: boolean; detail: string }[];
}

/** A validation's comparison.json (almasim.services.noise.compare.Comparison) for the page. */
export function verdict(comparison: unknown): Verdict | null {
	const c = comparison as {
		passed?: boolean;
		windows?: number;
		model?: { label?: string };
		metrics?: Record<string, Omit<MetricRow, 'metric' | 'within10'> & { within_10pct?: number }>;
		checks?: { name: string; passed: boolean; detail?: string }[];
	} | null;
	if (!c || typeof c.passed !== 'boolean') return null;
	return {
		passed: c.passed,
		windows: c.windows ?? 0,
		label: c.model?.label ?? '',
		metrics: Object.entries(c.metrics ?? {}).map(([metric, m]) => ({
			metric,
			kind: m.kind,
			n: m.n,
			median: m.median,
			q16: m.q16,
			q84: m.q84,
			p: m.p ?? null,
			within10: m.within_10pct ?? null
		})),
		checks: (c.checks ?? []).map((k) => ({
			name: k.name,
			passed: k.passed,
			detail: k.detail ?? ''
		}))
	};
}

/** The newest validation folder (validation/<model sha>) holding a comparison. */
export function newestValidation(
	entries: readonly { name: string; kind: string; modified: number }[]
): string | null {
	const folders = entries.filter((e) => e.kind === 'folder');
	if (!folders.length) return null;
	return folders.reduce((a, b) => (b.modified > a.modified ? b : a)).name;
}

/** A number as the tables show it: three significant digits. */
export function sig(value: number | null | undefined, digits = 3): string {
	if (value === null || value === undefined || !Number.isFinite(value)) return '—';
	return String(Number(value.toPrecision(digits)));
}
