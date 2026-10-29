import { fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { Values } from '#lib/utils/commandForm.js';

import NoisePicker from './NoisePicker.svelte';

const component = (name: string, enabled = true) => ({
	name,
	enabled,
	description: `${name} does things`
});

const listing = {
	site_folder: '/team/noise-models',
	models: [
		{
			spec: 'alma@1',
			name: 'alma',
			version: 1,
			newest: true,
			source: 'package',
			sha256: 'a'.repeat(64),
			description: 'Measured on real data.',
			components: [component('radiometer'), component('slow_phase'), component('fast_phase')],
			provenance: {
				method: 'fit',
				windows: 100,
				data: null,
				date: null,
				notes: null,
				validated: null
			}
		},
		{
			spec: 'team@1',
			name: 'team',
			version: 1,
			newest: true,
			source: 'site',
			sha256: 'b'.repeat(64),
			description: '',
			components: [component('radiometer'), component('slow_phase', false)],
			provenance: null
		}
	]
};

function values(noise_model = ''): Values {
	return { noise_model, noise_only: '', noise_without: '' };
}

describe('NoisePicker', () => {
	afterEach(() => vi.unstubAllGlobals());

	function serve() {
		vi.stubGlobal(
			'fetch',
			vi.fn(
				async () =>
					new Response(JSON.stringify(listing), {
						headers: { 'Content-Type': 'application/json' }
					})
			)
		);
	}

	it('offers no model, every listed model and a file', async () => {
		serve();
		render(NoisePicker, { values: values() });
		const select = screen.getByRole('combobox', { name: 'Noise model' });
		await screen.findByRole('option', { name: 'team@1 · team, newest' });
		expect([...select.querySelectorAll('option')].map((o) => o.textContent?.trim())).toEqual([
			'None: white thermal noise only',
			'alma@1 · ALMASim, newest',
			'team@1 · team, newest',
			'A model file (JSON)…'
		]);
		expect(screen.getByText(/Team models come from \/team\/noise-models/)).toBeTruthy();
	});

	it('writes the model and the components left out', async () => {
		serve();
		const form = $state(values());
		render(NoisePicker, { values: form });
		const select = screen.getByRole('combobox', { name: 'Noise model' });
		await screen.findByRole('option', { name: 'alma@1 · ALMASim, newest' });
		await fireEvent.change(select, { target: { value: 'alma@1' } });
		expect(form.noise_model).toBe('alma@1');
		expect(screen.getByText('Measured on real data.')).toBeTruthy();
		expect(screen.getByText('fit · 100 windows')).toBeTruthy();
		await fireEvent.click(screen.getByRole('checkbox', { name: /fast_phase/ }));
		expect(form).toMatchObject({ noise_only: '', noise_without: 'fast_phase' });
		await fireEvent.click(screen.getByRole('checkbox', { name: /slow_phase/ }));
		expect(form).toMatchObject({ noise_only: 'radiometer', noise_without: '' });
		expect(screen.getByText(/1 of 3 applied/)).toBeTruthy();
		// Another model starts from all of its components again.
		await fireEvent.change(select, { target: { value: 'team@1' } });
		expect(form).toMatchObject({ noise_model: 'team@1', noise_only: '', noise_without: '' });
		expect(screen.getByRole('checkbox', { name: /slow_phase/ })).toBeDisabled();
	});

	it('takes a file, and shows a run made with one as a file', async () => {
		serve();
		const form = $state(values('/data/mine.json'));
		render(NoisePicker, { values: form });
		await screen.findByRole('option', { name: 'alma@1 · ALMASim, newest' });
		expect(screen.getByRole('combobox', { name: 'Noise model' })).toHaveValue('file');
		expect(screen.getByDisplayValue('/data/mine.json')).toBeTruthy();
		await fireEvent.change(screen.getByRole('combobox', { name: 'Noise model' }), {
			target: { value: '' }
		});
		expect(form.noise_model).toBe('');
	});
});
