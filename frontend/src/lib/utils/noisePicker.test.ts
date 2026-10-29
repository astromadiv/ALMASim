import { describe, expect, it } from 'vitest';

import {
	FILE_CHOICE,
	applied,
	choiceOf,
	componentOptions,
	modelLabel,
	provenanceLine,
	tickedBy,
	type NoiseModelInfo
} from './noisePicker';

const component = (name: string, enabled = true) => ({ name, enabled, description: name });

function model(spec: string, overrides: Partial<NoiseModelInfo> = {}): NoiseModelInfo {
	const [name, version] = spec.split('@');
	return {
		spec,
		name,
		version: Number(version),
		newest: true,
		source: 'package',
		sha256: 'f'.repeat(64),
		description: '',
		components: ['radiometer', 'channel_correlation', 'antenna_gains', 'slow_phase'].map((n) =>
			component(n)
		),
		provenance: null,
		...overrides
	};
}

const alma = model('alma@2');
const models = [alma, model('alma@1', { newest: false }), model('team@1', { source: 'site' })];

describe('the noise model picker', () => {
	it('names each model, whose it is and whether it is the newest', () => {
		expect(models.map(modelLabel)).toEqual([
			'alma@2 · ALMASim, newest',
			'alma@1 · ALMASim',
			'team@1 · team, newest'
		]);
	});

	it('reads a --noise-model value back as a choice', () => {
		expect(choiceOf('', models)).toBe('');
		expect(choiceOf('none', models)).toBe('');
		expect(choiceOf('alma@1', models)).toBe('alma@1');
		expect(choiceOf(' alma ', models)).toBe('alma@2'); // a bare name is its newest
		expect(choiceOf('/data/my-model.json', models)).toBe(FILE_CHOICE);
	});

	it('writes the shorter of --noise-only and --noise-without, nothing when all are ticked', () => {
		expect(componentOptions(alma, applied(alma))).toEqual({ noise_only: '', noise_without: '' });
		expect(componentOptions(alma, ['radiometer', 'channel_correlation', 'slow_phase'])).toEqual({
			noise_only: '',
			noise_without: 'antenna_gains'
		});
		expect(componentOptions(alma, ['radiometer'])).toEqual({
			noise_only: 'radiometer',
			noise_without: ''
		});
		expect(componentOptions(alma, [])).toEqual({
			noise_only: '',
			noise_without: 'radiometer,channel_correlation,antenna_gains,slow_phase'
		});
	});

	it('reads the ticks back from those options, only among what the model applies', () => {
		const partial = model('x@1', {
			components: [component('radiometer'), component('slow_phase', false)]
		});
		expect(applied(partial)).toEqual(['radiometer']);
		expect(tickedBy(alma, '', '')).toEqual(applied(alma));
		expect(tickedBy(alma, 'radiometer, slow_phase', '')).toEqual(['radiometer', 'slow_phase']);
		expect(tickedBy(alma, '', 'antenna_gains')).toEqual([
			'radiometer',
			'channel_correlation',
			'slow_phase'
		]);
		expect(tickedBy(partial, 'slow_phase', '')).toEqual([]);
	});

	it('says where a model came from in one line', () => {
		expect(provenanceLine(alma)).toBe('');
		const measured = model('team@1', {
			provenance: {
				method: 'reduce',
				windows: 12,
				data: 'cycle 11',
				date: '2026-10-29',
				notes: null,
				validated: '0123456789ab'
			}
		});
		expect(provenanceLine(measured)).toBe(
			'reduce · 12 windows · cycle 11 · 2026-10-29 · validated (0123456789ab)'
		);
	});
});
