<!--
	Image calibrated MeasurementSets: work out each window's imaging geometry
	(`image compute-parameters`), review it, choose the imaging options, then
	run `image image-from-ms` and `image clean-quality` as one pipeline. Each
	window's progress comes from its task markers. The runs live in the
	address (?params=<id>&image=<id>&quality=<id>); ?csv=<path> starts from a
	parameters CSV made before.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { api, type Run } from '#lib/api/v2/client.js';
	import DashboardButton from '#lib/components/jobs/DashboardButton.svelte';
	import StateChip from '#lib/components/jobs/StateChip.svelte';
	import CommandSections from '#lib/components/run/CommandSections.svelte';
	import ErrorMessage from '#lib/components/shared/ErrorMessage.svelte';
	import PathPicker from '#lib/components/shared/PathPicker.svelte';
	import Spinner from '#lib/components/shared/Spinner.svelte';
	import { jobs } from '#lib/stores/jobs.svelte.js';
	import { toasts } from '#lib/toasts.svelte.js';
	import { toRecords } from '#lib/utils/archive.js';
	import { buildArgv, initialValues, type Command, type Values } from '#lib/utils/commandForm.js';
	import { errorMessage } from '#lib/utils/errors.js';
	import {
		PARAMETERS_CSV,
		QUALITY_COLUMNS,
		QUALITY_CSV,
		countStates,
		imagingArgv,
		parametersArgv,
		qualityArgv,
		qualityValue,
		shortMs,
		taskOf,
		taskState,
		type TaskState
	} from '#lib/utils/imaging.js';

	const PROGRESS_MS = 10_000;

	let imaging = $state<Command | null>(null);
	let values = $state<Values>({});
	let loadError = $state<unknown>(null);
	let busy = $state(false);

	let folder = $state('');
	let pattern = $state('*.cal');
	let allSpws = $state(false);
	let openCsv = $state('');
	let outputRoot = $state('');

	const paramsId = $derived(page.url.searchParams.get('params'));
	const imageId = $derived(page.url.searchParams.get('image'));
	const qualityId = $derived(page.url.searchParams.get('quality'));
	const csvParam = $derived(page.url.searchParams.get('csv'));

	let fetched = $state<Record<string, Run>>({});
	const runOf = (id: string | null) => (id ? (jobs.get(id) ?? fetched[id]) : undefined);
	const paramsRun = $derived(runOf(paramsId));
	const imageRun = $derived(runOf(imageId));
	const qualityRun = $derived(runOf(qualityId));

	const parametersCsv = $derived(
		csvParam ?? (paramsRun?.state === 'succeeded' ? `${paramsRun.run_dir}/${PARAMETERS_CSV}` : null)
	);
	/** The output root the image run uses (its fourth argument), else the form's. */
	const imagedRoot = $derived(imageRun?.argv[3] ?? null);

	let rows = $state<Record<string, string>[] | null>(null);
	let rowsError = $state<unknown>(null);
	let listings = $state<Record<string, Set<string>>>({});
	let quality = $state<{ columns: string[]; records: Record<string, string>[] } | null>(null);
	const qualityColumns = $derived(
		quality ? Object.keys(QUALITY_COLUMNS).filter((c) => quality?.columns.includes(c)) : []
	);

	const tasks = $derived(
		rows && (imagedRoot || outputRoot) ? rows.map((r) => taskOf(r, imagedRoot ?? outputRoot)) : []
	);
	const states = $derived(
		tasks.map((t): TaskState =>
			imageRun ? taskState(t, listings[t.msDir] ?? new Set()) : 'waiting'
		)
	);
	const counts = $derived(countStates(states));

	function go(params: Record<string, string | null>) {
		const next = new URLSearchParams(page.url.searchParams.toString());
		for (const [key, value] of Object.entries(params)) {
			if (value === null) next.delete(key);
			else next.set(key, value);
		}
		const query = next.toString();
		// eslint-disable-next-line svelte/no-navigation-without-resolve -- the path is resolved; only the query string is appended
		void goto(`${resolve('/imaging')}${query ? `?${query}` : ''}`, { replaceState: true });
	}

	$effect(() => {
		void (async () => {
			try {
				const { data } = await api.GET('/api/v2/commands');
				imaging = data?.find((c) => c.name === 'image image-from-ms') ?? null;
				if (imaging) values = initialValues(imaging);
			} catch (err) {
				loadError = err;
			}
		})();
	});

	$effect(() => {
		for (const id of [paramsId, imageId, qualityId]) {
			if (id && !jobs.get(id) && !fetched[id] && jobs.loaded) {
				void api
					.GET('/api/v2/jobs/{run_id}', { params: { path: { run_id: id } } })
					.then(({ data }) => data && (fetched = { ...fetched, [id]: data }))
					.catch((err) => (loadError = err));
			}
		}
	});

	async function readCsv(path: string) {
		const { data } = await api.GET('/api/v2/files/csv', {
			params: { query: { path, start: 0, rows: 5000 } }
		});
		return {
			columns: data?.columns ?? [],
			records: toRecords(data?.columns ?? [], data?.rows ?? [])
		};
	}

	$effect(() => {
		const path = parametersCsv;
		rows = null;
		rowsError = null;
		if (!path) return;
		let stale = false;
		readCsv(path).then(
			(read) => !stale && (rows = read.records),
			(err) => !stale && (rowsError = err)
		);
		return () => (stale = true);
	});

	// Each window's progress: list the MS folders while imaging runs, and once after.
	$effect(() => {
		const run = imageRun;
		const folders = [...new Set(tasks.map((t) => t.msDir))];
		if (!run || !folders.length) return;
		let stopped = false;
		let timer: ReturnType<typeof setTimeout> | undefined;
		async function look() {
			const next: Record<string, Set<string>> = {};
			for (const dir of folders) {
				try {
					const { data } = await api.GET('/api/v2/files/list', {
						params: { query: { path: dir } }
					});
					next[dir] = new Set(data?.entries.map((e) => e.name) ?? []);
				} catch {
					next[dir] = new Set(); // not made yet
				}
			}
			if (stopped) return;
			listings = next;
			if (!run?.finished) timer = setTimeout(look, PROGRESS_MS);
		}
		void look();
		return () => {
			stopped = true;
			clearTimeout(timer);
		};
	});

	$effect(() => {
		quality = null;
		if (qualityRun?.state !== 'succeeded' || !imagedRoot) return;
		let stale = false;
		readCsv(`${imagedRoot.replace(/\/+$/, '')}/${QUALITY_CSV}`).then(
			(read) => !stale && (quality = read),
			() => !stale && (quality = null)
		);
		return () => (stale = true);
	});

	async function submit(body: { argv: string[]; title: string; after?: string[] }) {
		const { data } = await api.POST('/api/v2/jobs', { body });
		if (data) jobs.put(data);
		return data ?? null;
	}

	async function computeParameters(event: SubmitEvent) {
		event.preventDefault();
		busy = true;
		try {
			const run = await submit({
				argv: parametersArgv(folder.trim(), pattern, allSpws),
				title: `Imaging parameters · ${folder.trim().split('/').pop()}`
			});
			if (run) {
				if (!outputRoot) outputRoot = `${folder.trim().replace(/\/+$/, '')}-images`;
				go({ params: run.id, csv: null, image: null, quality: null });
			}
		} catch (err) {
			toasts.show(`Could not start it: ${errorMessage(err)}`, 'danger');
		} finally {
			busy = false;
		}
	}

	async function startImaging() {
		if (!imaging || !parametersCsv || !outputRoot.trim()) return;
		busy = true;
		try {
			const options = buildArgv(imaging, values).slice(2);
			const out = outputRoot.trim();
			const image = await submit({
				argv: imagingArgv(parametersCsv, out, options),
				title: `Imaging · ${rows?.length ?? ''} windows`
			});
			if (!image) return;
			const check = await submit({
				argv: qualityArgv(out),
				title: 'Clean quality',
				after: [image.id]
			});
			go({ image: image.id, quality: check?.id ?? null });
			toasts.show('Imaging started; clean-quality runs when it succeeds.', 'ok');
		} catch (err) {
			toasts.show(`Could not start imaging: ${errorMessage(err)}`, 'danger');
		} finally {
			busy = false;
		}
	}

	const TILE: Record<TaskState, string> = {
		done: 'bg-ok-soft text-ok',
		running: 'bg-info text-white',
		failed: 'bg-danger text-white',
		waiting: 'bg-sunk text-fg-muted'
	};
	const fmt = (n: number, digits = 1) => (Number.isFinite(n) ? n.toFixed(digits) : '—');
	const step = (n: number, done: boolean) =>
		`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-sm font-bold text-white ${done ? 'bg-ok' : 'bg-navy'}`;
</script>

<svelte:head><title>Imaging · ALMASim</title></svelte:head>

<div class="flex flex-col gap-5">
	<div class="flex flex-col gap-1">
		<p class="alma-kicker">Workflow</p>
		<h1 class="m-0 text-3xl">Imaging</h1>
		<p class="m-0 text-fg-muted">
			Image calibrated MeasurementSets with WSClean, window by window, then measure how well each
			was cleaned. Imaging runs as a Slurm driver with its own workers.
		</p>
	</div>

	{#if loadError}<ErrorMessage error={loadError} title="Something did not load" />{/if}

	<!-- 1. Parameters -->
	<section class="alma-panel flex flex-col gap-4 rounded-lg p-5" aria-labelledby="i1">
		<div class="flex flex-wrap items-center gap-3">
			<span class={step(1, !!rows)}>1</span>
			<h2 id="i1" class="m-0 text-xl font-semibold">Measurement sets</h2>
			{#if paramsRun}
				<StateChip state={paramsRun.state} />
				<a class="text-sm" href={resolve('/jobs/[id]', { id: paramsRun.id })}
					>job {paramsRun.short_id}</a
				>
			{/if}
			{#if paramsId || csvParam}
				<button
					type="button"
					class="btn-ghost ml-auto px-3 py-1.5"
					onclick={() => go({ params: null, csv: null, image: null, quality: null })}
					>Start over</button
				>
			{/if}
		</div>
		{#if !paramsId && !csvParam}
			<form class="flex flex-col gap-4" onsubmit={computeParameters}>
				<PathPicker
					bind:value={folder}
					label="Folder of calibrated MeasurementSets"
					required
					hint="Every MS in it matching the pattern is imaged, one task per spectral window and field."
				/>
				<div class="flex flex-wrap items-end gap-4">
					<label class="flex flex-col gap-1 text-sm">
						<span class="font-medium">Pattern</span>
						<input class="w-40 rounded-md border px-3 py-2 font-mono" bind:value={pattern} />
					</label>
					<label class="flex items-center gap-2 pb-2 text-sm">
						<input type="checkbox" bind:checked={allSpws} /> Every window and field, calibrators too
					</label>
					<button type="submit" class="btn-primary ml-auto" disabled={busy || !folder.trim()}
						>Work out the imaging parameters</button
					>
				</div>
			</form>
			<details class="text-sm">
				<summary class="cursor-pointer text-fg-muted">Or start from a parameters CSV</summary>
				<div class="mt-3 flex flex-wrap items-end gap-3">
					<div class="min-w-72 flex-1">
						<PathPicker bind:value={openCsv} label="Parameters CSV" kind="file" />
					</div>
					<button
						type="button"
						class="btn-secondary"
						disabled={!openCsv.trim()}
						onclick={() => go({ csv: openCsv.trim(), params: null, image: null, quality: null })}
						>Open</button
					>
				</div>
			</details>
		{:else if paramsRun && !paramsRun.finished}
			<p class="m-0 flex items-center gap-2 text-sm text-fg-muted">
				<Spinner /> Reading the measurement sets{paramsRun.reason ? ` (${paramsRun.reason})` : ''}…
			</p>
		{:else if paramsRun && paramsRun.state !== 'succeeded'}
			<ErrorMessage
				error={`compute-parameters ${paramsRun.state}: ${paramsRun.reason || 'see its log.'}`}
				title="No parameters"
			/>
		{:else if rowsError}
			<ErrorMessage error={rowsError} title="Could not read the parameters" />
		{:else if !rows}
			<p class="flex items-center gap-2 text-sm text-fg-muted">
				<Spinner /> Reading the parameters…
			</p>
		{:else}
			<div class="overflow-x-auto">
				<table class="w-full text-sm">
					<thead>
						<tr class="text-left text-xs font-semibold tracking-wide text-fg-muted uppercase">
							<th scope="col" class="px-3 py-2">State</th>
							<th scope="col" class="px-3 py-2">Measurement set</th>
							<th scope="col" class="px-3 py-2">Task</th>
							<th scope="col" class="px-3 py-2">Source</th>
							<th scope="col" class="px-3 py-2 text-right">GHz</th>
							<th scope="col" class="px-3 py-2 text-right">Channels</th>
							<th scope="col" class="px-3 py-2 text-right">Beam ″</th>
							<th scope="col" class="px-3 py-2 text-right">FOV ″</th>
							<th scope="col" class="px-3 py-2 text-right">Minutes</th>
						</tr>
					</thead>
					<tbody>
						{#each rows as row, i (i)}
							{@const task = taskOf(row, imagedRoot ?? (outputRoot || '/'))}
							<tr class="border-t border-sunk">
								<td class="px-3 py-1.5">
									<span
										class="rounded px-2 py-0.5 text-xs font-semibold {TILE[states[i] ?? 'waiting']}"
										>{states[i] ?? 'waiting'}</span
									>
								</td>
								<td class="max-w-56 truncate px-3 py-1.5 font-mono text-xs" title={row.filename}
									>{task.msDir.split('/').pop()}</td
								>
								<td class="px-3 py-1.5 font-mono text-xs">{task.name}</td>
								<td class="max-w-48 truncate px-3 py-1.5" title={task.source}>{task.source}</td>
								<td class="px-3 py-1.5 text-right font-mono">{fmt(task.frequencyGhz, 3)}</td>
								<td class="px-3 py-1.5 text-right font-mono">{task.channels}</td>
								<td class="px-3 py-1.5 text-right font-mono">{fmt(task.beamArcsec, 2)}</td>
								<td class="px-3 py-1.5 text-right font-mono">{fmt(task.fovArcsec, 1)}</td>
								<td class="px-3 py-1.5 text-right font-mono">{fmt(task.minutes, 0)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</section>

	<!-- 2. Options and start -->
	{#if rows?.length && !imageId}
		<section class="flex flex-col gap-4" aria-labelledby="i2">
			<div class="alma-panel flex flex-col gap-4 rounded-lg p-5">
				<div class="flex items-center gap-3">
					<span class={step(2, false)}>2</span>
					<h2 id="i2" class="m-0 text-xl font-semibold">Imaging options</h2>
				</div>
				<PathPicker
					bind:value={outputRoot}
					label="Output folder"
					allowCreate
					required
					hint="Each MS gets a folder here, with one folder per window inside."
				/>
			</div>
			{#if imaging}
				<CommandSections command={imaging} bind:values omit={['Inputs']} />
			{/if}
			<div class="flex flex-wrap items-center gap-3">
				<button
					type="button"
					class="btn-accent"
					disabled={busy || !outputRoot.trim() || !imaging}
					onclick={startImaging}>Image {rows.length} window{rows.length === 1 ? '' : 's'}</button
				>
				<p class="m-0 text-xs text-fg-muted">
					Then <code>clean-quality</code> runs on the results, once imaging succeeds.
				</p>
			</div>
		</section>
	{/if}

	<!-- 3. Progress -->
	{#if imageId}
		<section class="alma-panel flex flex-col gap-4 rounded-lg p-5" aria-labelledby="i3">
			<div class="flex flex-wrap items-center gap-3">
				<span class={step(3, imageRun?.state === 'succeeded')}>3</span>
				<h2 id="i3" class="m-0 text-xl font-semibold">Imaging</h2>
				{#if imageRun}
					<StateChip state={imageRun.state} />
					<a class="text-sm" href={resolve('/jobs/[id]', { id: imageRun.id })}
						>job {imageRun.short_id}</a
					>
				{/if}
				<span class="text-sm text-fg-muted">
					{counts.done} done · {counts.running} running · {counts.failed} failed · {counts.waiting} waiting
				</span>
				{#if imageRun}
					<span class="ml-auto"
						><DashboardButton runId={imageRun.id} finished={imageRun.finished} /></span
					>
				{/if}
			</div>
			{#if tasks.length}
				<div
					role="list"
					aria-label="Windows"
					class="grid grid-cols-[repeat(auto-fill,minmax(11rem,1fr))] gap-1.5"
				>
					{#each tasks as task, i (`${task.msDir}/${task.name}`)}
						<div role="listitem" class="min-w-0">
							<!-- eslint-disable svelte/no-navigation-without-resolve -- the path is resolved; only the query string is appended -->
							<a
								href={`${resolve('/products')}?dir=${encodeURIComponent(`${task.msDir}/${task.name}`)}`}
								title="{task.msDir.split('/').pop()} {task.name}: {states[i]}; open its images"
								class="block truncate rounded px-2 py-1.5 font-mono text-xs font-semibold no-underline {TILE[
									states[i]
								]}"
							>
								{shortMs(task.msDir)}
								{task.name}
							</a>
							<!-- eslint-enable svelte/no-navigation-without-resolve -->
						</div>
					{/each}
				</div>
			{/if}
			<div class="flex flex-wrap items-center gap-3 border-t border-line pt-4">
				<h3 class="m-0 text-base font-semibold">Clean quality</h3>
				{#if qualityRun}
					<StateChip state={qualityRun.state} />
					<a class="text-sm" href={resolve('/jobs/[id]', { id: qualityRun.id })}
						>job {qualityRun.short_id}</a
					>
					{#if qualityRun.state === 'pending'}
						<span class="text-sm text-fg-muted">runs when imaging succeeds</span>
					{/if}
				{/if}
			</div>
			{#if quality && quality.records.length}
				<div class="overflow-x-auto">
					<table class="text-xs">
						<thead>
							<tr class="text-left text-fg-muted">
								<th scope="col" class="px-2 py-1 whitespace-nowrap">Task</th>
								{#each qualityColumns as column (column)}
									<th scope="col" class="px-2 py-1 text-right whitespace-nowrap"
										>{QUALITY_COLUMNS[column]}</th
									>
								{/each}
							</tr>
						</thead>
						<tbody>
							{#each quality.records as record, i (i)}
								<tr class="border-t border-sunk">
									<td class="px-2 py-1 font-mono whitespace-nowrap" title={record.task}
										>{shortMs(record.task.split('/')[0])} {record.task.split('/').pop()}</td
									>
									{#each qualityColumns as column (column)}
										<td
											class="px-2 py-1 text-right font-mono whitespace-nowrap"
											title={record[column]}>{qualityValue(record[column])}</td
										>
									{/each}
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</section>
	{/if}
</div>
