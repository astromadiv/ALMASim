/**
 * The archive workflow's three runs (query, resolve, download), as command
 * lines; and what the page reads back from their CSV files.
 */
import { RUN_DIR } from './runDir';

export const RESULTS_CSV = 'results.csv';
export const PRODUCTS_CSV = 'products.csv';

/** The query: the form's filters, its results saved in the run's own folder. */
export function queryArgv(filters: string[]): string[] {
	return [...filters, '--save-csv', `${RUN_DIR}/${RESULTS_CSV}`, '--yes'];
}

/**
 * Resolve the chosen observations' products into the run's folder; with
 * `expandArchives`, one row per file inside each product and auxiliary tar.
 */
export function resolveArgv(memberOusUids: string[], expandArchives = false): string[] {
	const argv = [
		'products',
		'resolve',
		'--member-ous-uid',
		memberOusUids.join(','),
		'--save-products-csv',
		`${RUN_DIR}/${PRODUCTS_CSV}`
	];
	if (expandArchives) argv.push('--expand-archives');
	return argv;
}

export interface DownloadChoice {
	productsCsv: string;
	destination: string;
	/** Where the raw and calibrated MeasurementSets go; empty: <destination>/archive_ms. */
	msFolder?: string;
	productFilter: string;
	/** Filename globs: only files matching one of them (all files when empty). */
	include?: string[];
	/** Filename globs: files matching one of them are skipped. */
	exclude?: string[];
	extract: boolean;
	unpack: boolean;
	calibrate: boolean;
}

/** Download the resolved products; unpacking and calibration follow on Slurm. */
export function downloadArgv(choice: DownloadChoice): string[] {
	const argv = [
		'products',
		'download',
		'--products-csv',
		choice.productsCsv,
		'--destination',
		choice.destination,
		'--product-filter',
		choice.productFilter
	];
	for (const pattern of choice.include ?? []) argv.push('--include', pattern);
	for (const pattern of choice.exclude ?? []) argv.push('--exclude', pattern);
	// Each step needs the one before it.
	if (choice.extract || choice.unpack || choice.calibrate) argv.push('--extract-tar');
	if (choice.unpack || choice.calibrate) argv.push('--unpack-ms');
	if (choice.calibrate) argv.push('--generate-calibrated-visibilities');
	if ((choice.unpack || choice.calibrate) && choice.msFolder?.trim()) {
		argv.push('--archive-output-root', choice.msFolder.trim());
	}
	return argv;
}

/** Where a download puts its MeasurementSets unless told otherwise (the CLI's default). */
export function defaultMsFolder(destination: string): string {
	const dest = destination.trim().replace(/\/+$/, '');
	return dest ? `${dest}/archive_ms` : '';
}

/** The calibrated-MS folder a `products download` command line writes, or null if it does not calibrate. */
export function calibratedMsFolder(argv: readonly string[]): string | null {
	if (argv[0] !== 'products' || argv[1] !== 'download') return null;
	if (!argv.includes('--generate-calibrated-visibilities')) return null;
	const valueOf = (flag: string) => {
		const i = argv.indexOf(flag);
		return i >= 0 ? (argv[i + 1] ?? null) : null;
	};
	const root = valueOf('--archive-output-root') ?? defaultMsFolder(valueOf('--destination') ?? '');
	return root ? `${root.replace(/\/+$/, '')}/calibrated_ms` : null;
}

/** CSV rows as records keyed by column. */
export function toRecords(columns: string[], rows: string[][]): Record<string, string>[] {
	return rows.map((row) => Object.fromEntries(columns.map((c, i) => [c, row[i] ?? ''])));
}

/** The distinct member OUS UIDs of the chosen rows, in order. */
export function memberOusUids(
	records: Record<string, string>[],
	chosen: Iterable<number>
): string[] {
	const seen = new Set<string>();
	for (const i of chosen) {
		const uid = records[i]?.member_ous_uid?.trim();
		if (uid) seen.add(uid);
	}
	return [...seen];
}

export interface ProductTotals {
	type: string;
	count: number;
	bytes: number;
}

/** Products per type (the download's --product-filter values), largest first. */
export function productTotals(records: Record<string, string>[]): ProductTotals[] {
	const byType = new Map<string, ProductTotals>();
	for (const record of records) {
		const type = record.product_type || 'other';
		const entry = byType.get(type) ?? { type, count: 0, bytes: 0 };
		entry.count += 1;
		entry.bytes += Number(record.content_length) || 0;
		byType.set(type, entry);
	}
	return [...byType.values()].sort((a, b) => b.bytes - a.bytes);
}

/** The comma-separated globs of a pattern field, trimmed, without empty ones. */
export function parsePatterns(text: string): string[] {
	return text
		.split(',')
		.map((p) => p.trim())
		.filter(Boolean);
}

/** A shell glob as a regular expression, as Python's fnmatch reads it (`*`, `?`, `[seq]`, `[!seq]`). */
function globRegExp(pattern: string): RegExp {
	let out = '';
	for (let i = 0; i < pattern.length; i++) {
		const c = pattern[i];
		if (c === '*') out += '.*';
		else if (c === '?') out += '.';
		else if (c === '[') {
			let j = i + 1;
			if (pattern[j] === '!') j++;
			if (pattern[j] === ']') j++;
			while (j < pattern.length && pattern[j] !== ']') j++;
			if (j >= pattern.length) out += '\\[';
			else {
				let body = pattern.slice(i + 1, j).replace(/\\/g, '\\\\');
				if (body.startsWith('!')) body = '^' + body.slice(1);
				else if (body.startsWith('^')) body = '\\' + body;
				out += `[${body}]`;
				i = j;
			}
		} else out += c.replace(/[.+^${}()|[\]\\]/g, '\\$&');
	}
	return new RegExp(`^${out}$`, 's');
}

/** Whether a filename passes the include globs (any; all files when none) and no exclude glob. */
export function matchesPatterns(name: string, include: string[], exclude: string[]): boolean {
	const hit = (patterns: string[]) => patterns.some((p) => globRegExp(p).test(name));
	return (include.length === 0 || hit(include)) && !hit(exclude);
}

/** The files and bytes a download picks: its product filter, then the filename globs. */
export function selection(
	records: Record<string, string>[],
	filter: string,
	include: string[] = [],
	exclude: string[] = []
): { count: number; bytes: number } {
	let count = 0;
	let bytes = 0;
	for (const record of records) {
		if (filter !== 'all' && (record.product_type || 'other') !== filter) continue;
		if (!matchesPatterns(record.filename ?? '', include, exclude)) continue;
		count += 1;
		bytes += Number(record.content_length) || 0;
	}
	return { count, bytes };
}
