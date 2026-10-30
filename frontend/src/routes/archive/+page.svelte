<!--
	Search the ALMA archive, choose observations, and download them, as three
	console runs: a metadata query, a product resolve, and a download that
	unpacks and calibrates on Slurm. The runs are in the address
	(?query=<id>&resolve=<id>), so leaving the page loses nothing; a saved
	results CSV opens with ?csv=<path>.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { api, type Run } from '#lib/api/v2/client.js';
	import ResultsTable from '#lib/components/archive/ResultsTable.svelte';
	import StateChip from '#lib/components/jobs/StateChip.svelte';
	import CommandSections from '#lib/components/run/CommandSections.svelte';
	import ErrorMessage from '#lib/components/shared/ErrorMessage.svelte';
	import PathPicker from '#lib/components/shared/PathPicker.svelte';
	import Spinner from '#lib/components/shared/Spinner.svelte';
	import { jobs } from '#lib/stores/jobs.svelte.js';
	import { toasts } from '#lib/toasts.svelte.js';
	import {
		PRODUCTS_CSV,
		RESULTS_CSV,
		defaultMsFolder,
		downloadArgv,
		memberOusUids,
		parsePatterns,
		productTotals,
		queryArgv,
		resolveArgv,
		selection,
		toRecords
	} from '#lib/utils/archive.js';
	import { buildArgv, initialValues, type Command, type Values } from '#lib/utils/commandForm.js';
	import { errorMessage } from '#lib/utils/errors.js';
	import { formatBytes } from '#lib/utils/format.js';

	const CSV_PAGE = 5000;

	let metadata = $state<Command | null>(null);
	let values = $state<Values>({});
	let loadError = $state<unknown>(null);
	let busy = $state(false);

	const queryId = $derived(page.url.searchParams.get('query'));
	const resolveId = $derived(page.url.searchParams.get('resolve'));
	const csvParam = $derived(page.url.searchParams.get('csv'));

	/** Runs not (or no longer) in the live store, fetched once. */
	let fetched = $state<Record<string, Run>>({});
	const runOf = (id: string | null) => (id ? (jobs.get(id) ?? fetched[id]) : undefined);
	const queryRun = $derived(runOf(queryId));
	const resolveRun = $derived(runOf(resolveId));

	let results = $state<{ columns: string[]; records: Record<string, string>[] } | null>(null);
	let resultsError = $state<unknown>(null);
	let selected = $state(new Set<number>());
	let products = $state<Record<string, string>[] | null>(null);
	let productsError = $state<unknown>(null);

	let openCsv = $state('');
	let destination = $state('');
	// Follows the destination (<destination>/archive_ms) until it is changed by hand.
	let msFolder = $state('');
	let msFolderDefault = '';
	let productFilter = $state('all');
	let listFiles = $state(false);
	let includeText = $state('');
	let excludeText = $state('');
	let extract = $state(true);
	let unpack = $state(true);
	let calibrate = $state(true);
	let space = $state<{ free_bytes: number; checked: string } | null>(null);

	const resultsCsv = $derived(
		csvParam ?? (queryRun?.state === 'succeeded' ? `${queryRun.run_dir}/${RESULTS_CSV}` : null)
	);
	const productsCsv = $derived(
		resolveRun?.state === 'succeeded' ? `${resolveRun.run_dir}/${PRODUCTS_CSV}` : null
	);
	const uids = $derived(results ? memberOusUids(results.records, selected) : []);
	const totals = $derived(products ? productTotals(products) : []);
	const include = $derived(parsePatterns(includeText));
	const exclude = $derived(parsePatterns(excludeText));
	const picked = $derived(
		products ? selection(products, productFilter, include, exclude) : { count: 0, bytes: 0 }
	);
	const downloadBytes = $derived(picked.bytes);
	/** The ticked rows, opened on the Simulate page instead. */
	const simulateHref = $derived(
		`${resolve('/simulate')}?metadata=${encodeURIComponent(resultsCsv ?? '')}&rows=${[...selected]
			.sort((a, b) => a - b)
			.join(',')}`
	);

	function go(params: Record<string, string | null>) {
		const next = new URLSearchParams(page.url.searchParams.toString());
		for (const [key, value] of Object.entries(params)) {
			if (value === null) next.delete(key);
			else next.set(key, value);
		}
		const query = next.toString();
		// eslint-disable-next-line svelte/no-navigation-without-resolve -- the path is resolved; only the query string is appended
		void goto(`${resolve('/archive')}${query ? `?${query}` : ''}`, { replaceState: true });
	}

	$effect(() => {
		void (async () => {
			try {
				const { data } = await api.GET('/api/v2/commands');
				metadata = data?.find((c) => c.name === 'metadata') ?? null;
				if (metadata) values = initialValues(metadata);
				const roots = await api.GET('/api/v2/files/roots');
				const first = roots.data?.roots.find((r) => r.exists && !r.runs);
				if (first && !destination) destination = `${first.path.replace(/\/$/, '')}/almasim-archive`;
			} catch (err) {
				loadError = err;
			}
		})();
	});

	// A run in the address that the live store does not follow: fetch it once.
	$effect(() => {
		for (const id of [queryId, resolveId]) {
			if (id && !jobs.get(id) && !fetched[id] && jobs.loaded) {
				void api
					.GET('/api/v2/jobs/{run_id}', { params: { path: { run_id: id } } })
					.then(({ data }) => data && (fetched = { ...fetched, [id]: data }))
					.catch((err) => (loadError = err));
			}
		}
	});

	async function readCsv(path: string) {
		let start = 0;
		let columns: string[] = [];
		const rows: string[][] = [];
		for (;;) {
			const { data } = await api.GET('/api/v2/files/csv', {
				params: { query: { path, start, rows: CSV_PAGE } }
			});
			if (!data) break;
			columns = data.columns;
			rows.push(...data.rows);
			start += data.rows.length;
			if (start >= data.total || data.rows.length === 0) break;
		}
		return { columns, records: toRecords(columns, rows) };
	}

	// The results, once there is a CSV to read.
	$effect(() => {
		const path = resultsCsv;
		results = null;
		resultsError = null;
		selected = new Set();
		if (!path) return;
		let stale = false;
		readCsv(path).then(
			(read) => !stale && (results = read),
			(err) => !stale && (resultsError = err)
		);
		return () => (stale = true);
	});

	// The resolved products.
	$effect(() => {
		const path = productsCsv;
		products = null;
		productsError = null;
		if (!path) return;
		let stale = false;
		readCsv(path).then(
			(read) => !stale && (products = read.records),
			(err) => !stale && (productsError = err)
		);
		return () => (stale = true);
	});

	$effect(() => {
		const next = defaultMsFolder(destination);
		if (msFolder === msFolderDefault) msFolder = next;
		msFolderDefault = next;
	});

	// Free space where the download goes, a moment after the folder changes.
	$effect(() => {
		const path = destination.trim();
		space = null;
		if (!path) return;
		const timer = setTimeout(async () => {
			try {
				const { data } = await api.GET('/api/v2/files/space', { params: { query: { path } } });
				space = data ?? null;
			} catch {
				space = null;
			}
		}, 500);
		return () => clearTimeout(timer);
	});

	async function submit(argv: string[], title: string): Promise<Run | null> {
		busy = true;
		try {
			const { data } = await api.POST('/api/v2/jobs', { body: { argv, title } });
			if (data) jobs.put(data);
			return data ?? null;
		} catch (err) {
			toasts.show(`Could not start it: ${errorMessage(err)}`, 'danger');
			return null;
		} finally {
			busy = false;
		}
	}

	async function search(event: SubmitEvent) {
		event.preventDefault();
		if (!metadata) return;
		const run = await submit(queryArgv(buildArgv(metadata, values)), 'Archive search');
		if (run) go({ query: run.id, csv: null, resolve: null });
	}

	async function resolveProducts() {
		const run = await submit(resolveArgv(uids, listFiles), `Resolve ${uids.length} observations`);
		if (run) go({ resolve: run.id });
	}

	async function download() {
		if (!productsCsv) return;
		const argv = downloadArgv({
			productsCsv,
			destination: destination.trim(),
			msFolder,
			productFilter,
			include,
			exclude,
			extract,
			unpack,
			calibrate
		});
		const run = await submit(argv, 'Archive download');
		if (run) {
			toasts.show('Download started; follow it on its page.', 'ok');
			await goto(resolve('/jobs/[id]', { id: run.id }));
		}
	}

	const step = (n: number, done: boolean) =>
		`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-sm font-bold ${done ? 'bg-ok text-white' : 'bg-navy text-white'}`;
</script>

<svelte:head><title>Archive · ALMASim</title></svelte:head>

<div class="flex flex-col gap-5">
	<div class="flex flex-col gap-1">
		<p class="alma-kicker">Workflow</p>
		<h1 class="m-0 text-3xl">ALMA archive</h1>
		<p class="m-0 text-fg-muted">
			Search the archive, choose observations, and download them. Each step is a job: queries and
			downloads run on Slurm, and unpacking and calibration follow there.
		</p>
	</div>

	{#if loadError}<ErrorMessage error={loadError} title="Something did not load" />{/if}

	<!-- 1. Search -->
	<section class="alma-panel flex flex-col gap-4 rounded-lg p-5" aria-labelledby="s1">
		<div class="flex flex-wrap items-center gap-3">
			<span class={step(1, !!results)}>1</span>
			<h2 id="s1" class="m-0 text-xl font-semibold">Find observations</h2>
			{#if queryRun}
				<StateChip state={queryRun.state} />
				<a href={resolve('/jobs/[id]', { id: queryRun.id })} class="text-sm"
					>job {queryRun.short_id}</a
				>
			{/if}
			{#if queryId || csvParam}
				<button
					type="button"
					class="btn-ghost ml-auto px-3 py-1.5"
					onclick={() => go({ query: null, csv: null, resolve: null })}>New search</button
				>
			{/if}
		</div>

		{#if !queryId && !csvParam}
			{#if metadata}
				<form class="flex flex-col gap-4" onsubmit={search}>
					<CommandSections command={metadata} bind:values omit={['Output']} />
					<div class="flex flex-wrap items-end gap-3">
						<button type="submit" class="btn-accent" disabled={busy}>Search the archive</button>
						<p class="m-0 text-xs text-fg-muted">
							Runs <code>almasim metadata</code> as a job; the results land in its run folder.
						</p>
					</div>
				</form>
				<details class="text-sm">
					<summary class="cursor-pointer text-fg-muted"
						>Or open a results CSV you saved before</summary
					>
					<div class="mt-3 flex flex-wrap items-end gap-3">
						<div class="min-w-72 flex-1">
							<PathPicker bind:value={openCsv} label="Results CSV" kind="file" />
						</div>
						<button
							type="button"
							class="btn-secondary"
							disabled={!openCsv.trim()}
							onclick={() => go({ csv: openCsv.trim(), query: null, resolve: null })}>Open</button
						>
					</div>
				</details>
			{:else if !loadError}
				<p class="flex items-center gap-2 text-sm text-fg-muted">
					<Spinner /> Loading the query form…
				</p>
			{/if}
		{:else if queryRun && !queryRun.finished}
			<p class="m-0 flex items-center gap-2 text-sm text-fg-muted">
				<Spinner /> Querying the archive{queryRun.reason ? ` (${queryRun.reason})` : ''}…
			</p>
		{:else if queryRun && queryRun.state !== 'succeeded'}
			<ErrorMessage
				error={`The query ${queryRun.state}: ${queryRun.reason || 'see its log.'}`}
				title="No results"
			/>
		{:else if csvParam}
			<p class="m-0 font-mono text-xs text-fg-muted">{csvParam}</p>
		{/if}
	</section>

	<!-- 2. Choose -->
	{#if resultsCsv}
		<section class="alma-panel flex flex-col gap-4 rounded-lg p-5" aria-labelledby="s2">
			<div class="flex flex-wrap items-center gap-3">
				<span class={step(2, !!resolveId)}>2</span>
				<h2 id="s2" class="m-0 text-xl font-semibold">Choose observations</h2>
			</div>
			{#if resultsError}
				<ErrorMessage error={resultsError} title="Could not read the results" />
			{:else if !results}
				<p class="flex items-center gap-2 text-sm text-fg-muted">
					<Spinner /> Reading the results…
				</p>
			{:else if results.records.length === 0}
				<p class="m-0 text-sm text-fg-muted">The query found nothing. Try fewer filters.</p>
			{:else}
				<ResultsTable records={results.records} columns={results.columns} bind:selected />
				<label class="flex items-center gap-2 text-sm"
					><input type="checkbox" bind:checked={listFiles} /> List the files inside each archive, to pick
					single files such as the cleaned images</label
				>
				<div class="flex flex-wrap items-center gap-3">
					<button
						type="button"
						class="btn-primary"
						disabled={!uids.length || busy}
						onclick={resolveProducts}
						>Find the products of {uids.length} observation{uids.length === 1 ? '' : 's'}</button
					>
					<p class="m-0 text-xs text-fg-muted">
						Asks the archive what each observation has to download (a short job).
					</p>
					{#if selected.size && resultsCsv}
						<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- the path is resolved; only the query string is appended -->
						<a href={simulateHref} class="btn-secondary ml-auto no-underline"
							>Simulate them instead</a
						>
					{/if}
				</div>
			{/if}
		</section>
	{/if}

	<!-- 3. Download -->
	{#if resolveId}
		<section class="alma-panel flex flex-col gap-4 rounded-lg p-5" aria-labelledby="s3">
			<div class="flex flex-wrap items-center gap-3">
				<span class={step(3, false)}>3</span>
				<h2 id="s3" class="m-0 text-xl font-semibold">Download</h2>
				{#if resolveRun}
					<StateChip state={resolveRun.state} />
					<a href={resolve('/jobs/[id]', { id: resolveRun.id })} class="text-sm"
						>job {resolveRun.short_id}</a
					>
				{/if}
			</div>
			{#if resolveRun && !resolveRun.finished}
				<p class="m-0 flex items-center gap-2 text-sm text-fg-muted">
					<Spinner /> Resolving the products{resolveRun.reason ? ` (${resolveRun.reason})` : ''}…
				</p>
			{:else if resolveRun && resolveRun.state !== 'succeeded'}
				<ErrorMessage
					error={`Resolving ${resolveRun.state}: ${resolveRun.reason || 'see its log.'}`}
					title="No products"
				/>
			{:else if productsError}
				<ErrorMessage error={productsError} title="Could not read the products" />
			{:else if products}
				<div class="flex flex-wrap gap-6">
					<table class="self-start text-sm">
						<thead>
							<tr class="text-left text-xs text-fg-muted uppercase">
								<th scope="col" class="py-1 pr-6">Product type</th>
								<th scope="col" class="py-1 pr-6 text-right">Files</th>
								<th scope="col" class="py-1 text-right">Size</th>
							</tr>
						</thead>
						<tbody>
							{#each totals as t (t.type)}
								<tr class="border-t border-sunk">
									<td class="py-1 pr-6">{t.type}</td>
									<td class="py-1 pr-6 text-right font-mono">{t.count}</td>
									<td class="py-1 text-right font-mono">{formatBytes(t.bytes)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
					<div class="flex min-w-72 flex-1 flex-col gap-4">
						<label class="flex flex-col gap-1 text-sm">
							<span class="font-medium">Download</span>
							<select class="rounded-md border px-3 py-2" bind:value={productFilter}>
								<option value="all">everything</option>
								{#each totals as t (t.type)}<option value={t.type}>{t.type} only</option>{/each}
							</select>
						</label>
						<label class="flex flex-col gap-1 text-sm">
							<span class="font-medium">Only files named</span>
							<input
								class="rounded-md border px-3 py-2 font-mono"
								placeholder="*.pbcor.fits"
								bind:value={includeText}
							/>
						</label>
						<label class="flex flex-col gap-1 text-sm">
							<span class="font-medium">Except files named</span>
							<input
								class="rounded-md border px-3 py-2 font-mono"
								placeholder="*_bp.*"
								bind:value={excludeText}
							/>
							<span class="text-xs text-fg-muted"
								>Comma-separated name patterns: <code>*</code> any text, <code>?</code> one character.
								They pick among the files inside the archives when the products were found with “list
								the files inside each archive”, and among whole archives otherwise.</span
							>
						</label>
						<PathPicker bind:value={destination} label="Destination" allowCreate required />
						<p class="m-0 text-sm" class:text-danger={space && space.free_bytes < downloadBytes}>
							{picked.count} file{picked.count === 1 ? '' : 's'}, {formatBytes(downloadBytes)} to download{space
								? ` · ${formatBytes(space.free_bytes)} free there`
								: ''}.
							{#if space && space.free_bytes < downloadBytes}<strong>Not enough space.</strong>{/if}
							{#if unpack || calibrate}
								<span class="text-fg-muted">Unpacking and calibrating need several times more.</span
								>
							{/if}
						</p>
						<fieldset class="flex flex-col gap-2 text-sm">
							<legend class="mb-1 font-medium">Then, on Slurm</legend>
							<label class="flex items-center gap-2"
								><input type="checkbox" bind:checked={extract} disabled={unpack || calibrate} /> Extract
								the tar archives</label
							>
							<label class="flex items-center gap-2"
								><input type="checkbox" bind:checked={unpack} disabled={calibrate} /> Make raw MeasurementSets</label
							>
							<label class="flex items-center gap-2"
								><input type="checkbox" bind:checked={calibrate} /> Restore the calibration (calibrated
								MeasurementSets)</label
							>
						</fieldset>
						{#if unpack || calibrate}
							<PathPicker
								bind:value={msFolder}
								label="MeasurementSets folder"
								allowCreate
								required
								hint="Raw MSs go to raw_ms/ in it and calibrated ones to calibrated_ms/; the Imaging page starts from the last download's calibrated_ms/."
							/>
						{/if}
						<div>
							<button
								type="button"
								class="btn-accent"
								disabled={busy ||
									!destination.trim() ||
									((unpack || calibrate) && !msFolder.trim()) ||
									downloadBytes === 0 ||
									(space !== null && space.free_bytes < downloadBytes)}
								onclick={download}>Download {formatBytes(downloadBytes)}</button
							>
						</div>
					</div>
				</div>
			{:else}
				<p class="flex items-center gap-2 text-sm text-fg-muted">
					<Spinner /> Reading the products…
				</p>
			{/if}
		</section>
	{/if}
</div>
