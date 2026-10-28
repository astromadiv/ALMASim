import { render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';

import DashboardButton from './DashboardButton.svelte';

/** A fake job endpoint whose `dashboard` flag a test can turn on. */
function fakeJob(state: { dashboard: boolean; finished?: boolean }) {
	const calls: string[] = [];
	vi.stubGlobal(
		'fetch',
		vi.fn(async (input: Request) => {
			calls.push(new URL(input.url).pathname);
			return new Response(
				JSON.stringify({ dashboard: state.dashboard, finished: state.finished ?? false }),
				{ headers: { 'Content-Type': 'application/json' } }
			);
		})
	);
	return calls;
}

describe('DashboardButton', () => {
	afterEach(() => vi.unstubAllGlobals());

	it('appears once the running job has a dashboard, linking to the relay', async () => {
		const state = { dashboard: false };
		const calls = fakeJob(state);
		render(DashboardButton, { runId: 'R1', finished: false, pollMs: 30 });
		await vi.waitFor(() => expect(calls.length).toBeGreaterThan(1));
		expect(screen.queryByRole('link', { name: 'Dask dashboard' })).toBeNull();
		state.dashboard = true;
		const link = await screen.findByRole('link', { name: 'Dask dashboard' });
		expect(link.getAttribute('href')).toMatch(/\/api\/v2\/jobs\/R1\/dashboard\/status$/);
		expect(link).toHaveAttribute('target', '_blank');
		expect(calls.every((path) => path.endsWith('/api/v2/jobs/R1'))).toBe(true);
	});

	it('asks nothing for a finished run, and stops asking when it finishes', async () => {
		const calls = fakeJob({ dashboard: false, finished: true });
		const view = render(DashboardButton, { runId: 'R2', finished: true, pollMs: 30 });
		await new Promise((r) => setTimeout(r, 100));
		expect(calls).toEqual([]);
		await view.rerender({ runId: 'R2', finished: false, pollMs: 30 });
		await vi.waitFor(() => expect(calls.length).toBe(1));
		await new Promise((r) => setTimeout(r, 150));
		expect(calls.length).toBe(1); // the job said it had finished
		expect(screen.queryByRole('link', { name: 'Dask dashboard' })).toBeNull();
	});
});
