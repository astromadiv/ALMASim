<!--
	A link to a running job's Dask dashboard, which the console relays
	(/api/v2/jobs/<id>/dashboard/). It shows once the job's scheduler has
	noted its dashboard, which can take a while after the job starts, so it
	asks every `pollMs` until then, and goes away when the run ends.
-->
<script lang="ts">
	import { api } from '#lib/api/v2/client.js';
	import { consoleUrl } from '#lib/config.js';

	interface Props {
		runId: string;
		finished: boolean;
		pollMs?: number;
	}

	let { runId, finished, pollMs = 15_000 }: Props = $props();

	let available = $state(false);

	$effect(() => {
		available = false;
		if (finished) return;
		const id = runId;
		let stopped = false;
		let timer: ReturnType<typeof setTimeout> | undefined;
		async function look() {
			try {
				const { data } = await api.GET('/api/v2/jobs/{run_id}', {
					params: { path: { run_id: id } }
				});
				if (stopped) return;
				available = data?.dashboard ?? false;
				if (data?.finished) return;
			} catch {
				// try again at the next tick
			}
			if (!stopped && !available) timer = setTimeout(look, pollMs);
		}
		void look();
		return () => {
			stopped = true;
			clearTimeout(timer);
		};
	});
</script>

{#if available && !finished}
	<!-- eslint-disable svelte/no-navigation-without-resolve -- the console's relay, not a page -->
	<a
		class="btn-secondary no-underline"
		href={`${consoleUrl()}/api/v2/jobs/${runId}/dashboard/status`}
		target="_blank"
		rel="noopener noreferrer">Dask dashboard</a
	>
	<!-- eslint-enable svelte/no-navigation-without-resolve -->
{/if}
