/**
 * The Simulate page's noise model: which model (`--noise-model`) and which of
 * its components (`--noise-only` / `--noise-without`), from the models
 * `/api/v2/noise-models` lists.
 */
import type { components } from '#lib/api/v2/schema.js';

export type NoiseModelInfo = components['schemas']['NoiseModelInfo'];

/** The picker's "a JSON file" choice; the file's path is the model then. */
export const FILE_CHOICE = 'file';

/** What the dropdown shows for a model: its spec, whose it is, and whether it is the newest. */
export function modelLabel(model: NoiseModelInfo): string {
	const whose = model.source === 'site' ? 'team' : 'ALMASim';
	return `${model.spec} · ${whose}${model.newest ? ', newest' : ''}`;
}

/** The dropdown's choice for a `--noise-model` value: a listed spec, none (''), or a file. */
export function choiceOf(value: string, models: NoiseModelInfo[]): string {
	const text = value.trim();
	if (!text || text === 'none') return '';
	const named = models.find(
		(m) => m.spec === text || (m.newest && m.name === text) // a bare name is its newest
	);
	return named ? named.spec : FILE_CHOICE;
}

/** The components a model applies (the ones it has switched on). */
export function applied(model: NoiseModelInfo): string[] {
	return model.components.filter((c) => c.enabled).map((c) => c.name);
}

/**
 * The component options for the ticked components of ``model``: nothing when
 * every applied component is ticked, else the shorter of `--noise-only` (what
 * stays) and `--noise-without` (what goes). Both give the same model.
 */
export function componentOptions(
	model: NoiseModelInfo,
	ticked: Iterable<string>
): { noise_only: string; noise_without: string } {
	const on = applied(model);
	const keep = new Set(ticked);
	const kept = on.filter((name) => keep.has(name));
	const dropped = on.filter((name) => !keep.has(name));
	if (!dropped.length) return { noise_only: '', noise_without: '' };
	if (kept.length && kept.length < dropped.length) {
		return { noise_only: kept.join(','), noise_without: '' };
	}
	return { noise_only: '', noise_without: dropped.join(',') };
}

/** The components ticked by a run's `--noise-only` / `--noise-without` values, for a model. */
export function tickedBy(model: NoiseModelInfo, only: string, without: string): string[] {
	const names = (text: string) =>
		text
			.split(',')
			.map((s) => s.trim())
			.filter(Boolean);
	const keep = only.trim() ? new Set(names(only)) : null;
	const drop = new Set(names(without));
	return applied(model).filter((name) => (keep === null || keep.has(name)) && !drop.has(name));
}

/** A model's provenance in one line: what it was measured on, and its validation. */
export function provenanceLine(model: NoiseModelInfo): string {
	const p = model.provenance;
	if (!p) return '';
	const parts = [p.method];
	if (p.windows) parts.push(`${p.windows} windows`);
	if (p.data) parts.push(p.data);
	if (p.date) parts.push(p.date);
	if (p.validated) parts.push(`validated (${p.validated})`);
	return parts.join(' · ');
}
