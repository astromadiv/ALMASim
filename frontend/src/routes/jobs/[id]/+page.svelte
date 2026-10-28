<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { api, type RunDetail } from '#lib/api/v2/client.js';
	import { consoleUrl } from '#lib/config.js';
	import { clock } from '#lib/clock.svelte.js';
	import DashboardButton from '#lib/components/jobs/DashboardButton.svelte';
	import LogView from '#lib/components/jobs/LogView.svelte';
	import StateChip from '#lib/components/jobs/StateChip.svelte';
	import Dialog from '#lib/components/shared/Dialog.svelte';
	import ErrorMessage from '#lib/components/shared/ErrorMessage.svelte';
	import Spinner from '#lib/components/shared/Spinner.svelte';
	import { jobs } from '#lib/stores/jobs.svelte.js';
	import { toasts } from '#lib/toasts.svelte.js';
	import { errorMessage } from '#lib/utils/errors.js';
	import { commandLine, elapsedLabel, formatDuration, whereLabel } from '#lib/utils/jobs.js';

	type Tab = 'log' | 'parameters' | 'script' | 'files';

	const id = $derived(page.params.id ?? '');
	let detail = $state<RunDetail | null>(null);
	let loadError = $state<unknown>(null);
	let tab = $state<Tab>('log');
	let confirmingCancel = $state(false);
	let confirmingDelete = $state(false);
	let busy = $state(false);

	$effect(() => clock.use());

	// The live state from the event stream wins over the fetched copy.
	const run = $derived.by(() => {
		const live = detail ? jobs.get(detail.id) : undefined;
		return detail && live ? { ...detail, ...live } : detail;
	});
	const isMine = $derived(run !== null && (jobs.user === null || run.owner === jobs.user));

	async function load(runId: string) {
		try {
			const { data } = await api.GET('/api/v2/jobs/{run_id}', {
				params: { path: { run_id: runId } }
			});
			detail = data ?? null;
			loadError = null;
		} catch (err) {
			loadError = err;
		}
	}

	$effect(() => {
		detail = null;
		void load(id);
	});

	// Fetch again when the run changes state (new files, exit code, host).
	let seenState = '';
	$effect(() => {
		const state = run?.state ?? '';
		if (detail && state && seenState && state !== seenState) void load(detail.id);
		seenState = state;
	});

	async function copyCommand() {
		if (!run) return;
		try {
			await navigator.clipboard.writeText(commandLine(run.argv));
			toasts.show('Command copied', 'ok');
		} catch {
			toasts.show(commandLine(run.argv), 'info', 15000);
		}
	}

	async function runAgain() {
		if (!run) return;
		busy = true;
		try {
			const { data } = await api.POST('/api/v2/jobs/{run_id}/resubmit', {
				params: { path: { run_id: run.id } }
			});
			if (data) {
				jobs.put(data);
				toasts.show(`Submitted again as ${data.short_id}`, 'ok');
				await goto(resolve('/jobs/[id]', { id: data.id }));
			}
		} catch (err) {
			toasts.show(`Could not run it again: ${errorMessage(err)}`, 'danger');
		} finally {
			busy = false;
		}
	}

	async function remove() {
		if (!run) return;
		busy = true;
		try {
			await api.DELETE('/api/v2/jobs/{run_id}', { params: { path: { run_id: run.id } } });
			jobs.remove(run.id);
			toasts.show(`Deleted ${run.short_id}`, 'ok');
			await goto(resolve('/jobs'));
		} catch (err) {
			toasts.show(`Could not delete it: ${errorMessage(err)}`, 'danger');
		} finally {
			busy = false;
			confirmingDelete = false;
		}
	}

	async function cancel() {
		if (!run) return;
		busy = true;
		try {
			const { data } = await api.POST('/api/v2/jobs/{run_id}/cancel', {
				params: { path: { run_id: run.id } }
			});
			if (data) jobs.put(data);
			toasts.show('Cancel requested', 'ok');
			confirmingCancel = false;
		} catch (err) {
			toasts.show(`Could not cancel: ${errorMessage(err)}`, 'danger');
		} finally {
			busy = false;
		}
	}

	const TABS: { value: Tab; label: string }[] = [
		{ value: 'log', label: 'Log' },
		{ value: 'parameters', label: 'Parameters' },
		{ value: 'script', label: 'Script' },
		{ value: 'files', label: 'Files' }
	];

	const time = (seconds: number | null | undefined) =>
		seconds == null ? '—' : new Date(seconds * 1000).toLocaleString();
	const fileHref = (dir: string, name: string) =>
		`${consoleUrl()}/api/v2/files/download?path=${encodeURIComponent(`${dir}/${name}`)}`;
</script>

<svelte:head
	><title>{run ? `${run.title || run.command} · Jobs` : 'Job'} · ALMASim</title></svelte:head
>

<div class="flex flex-col gap-5">
	<nav aria-label="Breadcrumb" class="text-sm text-fg-muted">
		<a href={resolve('/jobs')}>Jobs</a> ›
		<span class="font-mono">{run?.short_id ?? id}</span>
	</nav>

	{#if loadError && !run}
		<ErrorMessage error={loadError} title="Could not load this job" onRetry={() => load(id)} />
	{:else if !run}
		<p class="flex items-center gap-2 text-sm text-fg-muted"><Spinner /> Loading…</p>
	{:else}
		<header class="flex flex-wrap items-start gap-4">
			<div class="flex min-w-0 flex-1 basis-96 flex-col gap-2">
				<div class="flex flex-wrap items-center gap-2.5">
					<StateChip
						state={run.state}
						detail={run.started_at == null && run.finished
							? undefined
							: elapsedLabel(run, clock.now)}
					/>
					{#if run.reason}<span class="text-sm text-fg-muted">{run.reason}</span>{/if}
				</div>
				<h1 class="m-0 text-3xl break-words">{run.title || run.command}</h1>
				<p class="m-0 font-mono text-xs break-all text-fg-muted">{run.run_dir}</p>
			</div>
			<div class="flex flex-wrap gap-2">
				<DashboardButton runId={run.id} finished={run.finished} />
				<button type="button" class="btn-secondary" onclick={copyCommand}>Copy command</button>
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- the path is resolved; only the query string is appended -->
				<a class="btn-secondary no-underline" href={`${resolve('/run')}?from=${run.id}`}
					>Edit and run</a
				>
				{#if isMine}
					<button type="button" class="btn-secondary" disabled={busy} onclick={runAgain}
						>Run again</button
					>
					{#if !run.finished}
						<button
							type="button"
							class="btn-secondary border-danger text-danger"
							disabled={busy}
							onclick={() => (confirmingCancel = true)}>Cancel job</button
						>
					{:else}
						<button
							type="button"
							class="btn-secondary border-danger text-danger"
							disabled={busy}
							onclick={() => (confirmingDelete = true)}>Delete</button
						>
					{/if}
				{/if}
			</div>
		</header>

		{#if run.parents.length}
			<p class="m-0 text-sm text-fg-muted">
				Starts after
				{#each run.parents as parent, i (parent)}
					{#if i > 0},
					{/if}
					<a class="font-mono" href={resolve('/jobs/[id]', { id: parent })}
						>{jobs.get(parent)?.short_id ?? parent.slice(-10)}</a
					>
					{#if jobs.get(parent)}({jobs.get(parent)?.state}){/if}
				{/each}
			</p>
		{/if}

		<div class="flex flex-wrap items-start gap-6">
			<section class="alma-panel min-w-0 flex-[999_1_36rem] overflow-hidden rounded-lg">
				<div
					role="tablist"
					aria-label="Job views"
					class="flex gap-5 overflow-x-auto border-b border-line px-5"
				>
					{#each TABS as option (option.value)}
						<button
							type="button"
							role="tab"
							id="tab-{option.value}"
							aria-controls="panel-{option.value}"
							aria-selected={tab === option.value}
							class="border-b-2 border-transparent px-1 py-2.5 text-sm font-medium whitespace-nowrap text-fg-muted aria-selected:border-magenta aria-selected:font-bold aria-selected:text-navy"
							onclick={() => (tab = option.value)}>{option.label}</button
						>
					{/each}
				</div>
				<div role="tabpanel" id="panel-{tab}" aria-labelledby="tab-{tab}">
					{#if tab === 'log'}
						<LogView runId={run.id} runDir={run.run_dir} finished={run.finished} />
					{:else if tab === 'parameters'}
						<div class="flex flex-col gap-4 p-5">
							<div>
								<h2 class="mb-1 text-sm font-semibold text-navy">Command</h2>
								<pre
									class="m-0 rounded-md bg-sunk p-3 font-mono text-xs leading-5 break-all whitespace-pre-wrap">{commandLine(
										run.argv
									)}</pre>
								<p class="mt-1 text-xs text-fg-dim">
									As submitted, with what the console filled in for the cluster.
								</p>
							</div>
							{#if Object.keys(run.params).length}
								<div>
									<h2 class="mb-1 text-sm font-semibold text-navy">Form values</h2>
									<pre
										class="m-0 overflow-x-auto rounded-md bg-sunk p-3 font-mono text-xs leading-5">{JSON.stringify(
											run.params,
											null,
											2
										)}</pre>
								</div>
							{/if}
						</div>
					{:else if tab === 'script'}
						{#await run.executor === 'slurm' ? api.GET( '/api/v2/files/text', { params: { query: { path: `${run.run_dir}/job.sbatch` } } } ) : null}
							<p class="flex items-center gap-2 p-5 text-sm text-fg-muted">
								<Spinner /> Loading…
							</p>
						{:then script}
							{#if script?.data}
								<pre class="m-0 overflow-x-auto p-5 font-mono text-xs leading-5">{script.data
										.text}</pre>
							{:else}
								<p class="p-5 text-sm text-fg-muted">
									A console-server job has no batch script: the console starts
									<code>python -m almasim.jobs.wrapper</code> on its own machine, under nice and ionice.
								</p>
							{/if}
						{:catch err}
							<div class="p-5"><ErrorMessage error={err} title="Could not read job.sbatch" /></div>
						{/await}
					{:else}
						<p class="m-0 px-5 pt-4 text-sm">
							<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- the path is resolved; only the query string is appended -->
							<a href={`${resolve('/products')}?dir=${encodeURIComponent(run.run_dir)}`}
								>Open the run folder in Products</a
							>
							<span class="text-fg-muted">to look at its images and cubes.</span>
						</p>
						<ul class="m-0 flex list-none flex-col p-2">
							{#each detail?.files ?? [] as name (name)}
								<li class="flex items-center gap-3 rounded px-3 py-1.5 hover:bg-hover">
									<span class="min-w-0 flex-1 truncate font-mono text-sm">{name}</span>
									<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- an API download, not a page -->
									<a class="text-sm" href={fileHref(run.run_dir, name)} download>Download</a>
								</li>
							{:else}
								<li class="px-3 py-2 text-sm text-fg-muted">The run folder is empty.</li>
							{/each}
						</ul>
					{/if}
				</div>
			</section>

			<aside class="flex min-w-0 flex-[1_1_20rem] flex-col gap-4">
				<section class="alma-panel flex flex-col gap-3 rounded-lg p-5">
					<h2 class="m-0 text-lg font-semibold">Where</h2>
					<dl class="m-0 grid grid-cols-[8rem_minmax(0,1fr)] gap-x-3 gap-y-2 text-sm">
						<dt class="text-fg-muted">Runs on</dt>
						<dd class="m-0">{whereLabel(run)}</dd>
						<dt class="text-fg-muted">Resources</dt>
						<dd class="m-0">
							{run.resources.cpus} CPU · {run.resources.memory} · limit {run.resources.time}
						</dd>
						{#if run.resources.account}
							<dt class="text-fg-muted">Account</dt>
							<dd class="m-0">{run.resources.account}</dd>
						{/if}
						<dt class="text-fg-muted">Host</dt>
						<dd class="m-0 font-mono text-xs break-all">{run.host ?? '—'}</dd>
						<dt class="text-fg-muted">Owner</dt>
						<dd class="m-0">{run.owner}</dd>
						<dt class="text-fg-muted">Submitted</dt>
						<dd class="m-0">{time(run.created_at)}</dd>
						<dt class="text-fg-muted">Started</dt>
						<dd class="m-0">{time(run.started_at)}</dd>
						<dt class="text-fg-muted">Finished</dt>
						<dd class="m-0">{time(run.finished_at)}</dd>
						{#if run.exit_code != null || run.signal}
							<dt class="text-fg-muted">Exit</dt>
							<dd class="m-0">{run.signal ?? `code ${run.exit_code}`}</dd>
						{/if}
					</dl>
				</section>

				{#if run.executor === 'slurm' && run.slurm_job_id}
					<section class="alma-panel flex flex-col gap-3 rounded-lg p-5">
						<h2 class="m-0 text-lg font-semibold">Slurm accounting</h2>
						{#await api.GET( '/api/v2/jobs/{run_id}/accounting', { params: { path: { run_id: run.id } } } )}
							<p class="flex items-center gap-2 text-sm text-fg-muted"><Spinner /> sacct…</p>
						{:then acct}
							{#if acct.data?.rows.length}
								<div class="overflow-x-auto">
									<table class="w-full text-xs">
										<thead>
											<tr class="text-left text-fg-muted">
												<th scope="col" class="py-1 pr-3">Step</th>
												<th scope="col" class="py-1 pr-3">State</th>
												<th scope="col" class="py-1 pr-3">Elapsed</th>
												<th scope="col" class="py-1">Max RSS</th>
											</tr>
										</thead>
										<tbody>
											{#each acct.data.rows as row (row.JobID)}
												<tr class="border-t border-sunk">
													<td class="py-1 pr-3 font-mono">{row.JobID}</td>
													<td class="py-1 pr-3">{row.State}</td>
													<td class="py-1 pr-3 font-mono">{row.Elapsed}</td>
													<td class="py-1 font-mono">{row.MaxRSS || '—'}</td>
												</tr>
											{/each}
										</tbody>
									</table>
								</div>
								<p class="m-0 text-xs text-fg-dim">
									{acct.data.rows[0].NodeList || 'no node yet'} · {acct.data.rows[0].AllocCPUS} CPU ·
									asked {acct.data.rows[0].ReqMem}
								</p>
							{:else}
								<p class="m-0 text-sm text-fg-muted">Slurm has no record of it yet.</p>
							{/if}
						{:catch err}
							<ErrorMessage error={err} />
						{/await}
					</section>
				{/if}

				{#if run.state === 'failed' || run.state === 'lost'}
					<section class="flex flex-col gap-1.5 rounded-lg border border-danger bg-danger-soft p-4">
						<h2 class="m-0 text-base font-bold text-danger">It did not finish</h2>
						<p class="m-0 text-sm">
							{run.reason || 'See the log for why.'}
							{#if run.started_at && run.finished_at}
								It ran {formatDuration(run.finished_at - run.started_at)}.
							{/if}
						</p>
					</section>
				{/if}
			</aside>
		</div>
	{/if}
</div>

{#if confirmingCancel && run}
	<Dialog title="Cancel this job?" onClose={() => (confirmingCancel = false)}>
		<p class="m-0 text-sm">
			<strong>{run.title || run.command}</strong> ({run.short_id}) stops now{run.slurm_job_id
				? ` (scancel ${run.slurm_job_id})`
				: ''}. Runs that wait on it will not start. What it already wrote stays in its folders.
		</p>
		{#snippet footer()}
			<button type="button" class="btn-ghost" onclick={() => (confirmingCancel = false)}
				>Keep it running</button
			>
			<button type="button" class="btn-danger" disabled={busy} onclick={cancel}>Cancel job</button>
		{/snippet}
	</Dialog>
{/if}

{#if confirmingDelete && run}
	<Dialog title="Delete this run?" onClose={() => (confirmingDelete = false)}>
		<p class="m-0 text-sm">
			Its folder goes, with its logs, script and everything written there:
			<code class="break-all">{run.run_dir}</code>. Outputs it wrote elsewhere stay. This cannot be
			undone.
		</p>
		{#snippet footer()}
			<button type="button" class="btn-ghost" onclick={() => (confirmingDelete = false)}
				>Keep it</button
			>
			<button type="button" class="btn-danger" disabled={busy} onclick={remove}>Delete</button>
		{/snippet}
	</Dialog>
{/if}
