<!--
	Build a noise model from real data: list an imaging run's windows
	(`noise windows`), tick the ones to learn from, then measure them, fit
	their calibration residuals, reduce them into a candidate model and
	validate it against them, as one chain of jobs. The runs live in the
	address (?windows=<id>&chain=<id>,<id>,…).
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { api, type Run } from '#lib/api/v2/client.js';
	import DashboardButton from '#lib/components/jobs/DashboardButton.svelte';
	import StateChip from '#lib/components/jobs/StateChip.svelte';
	import ErrorMessage from '#lib/components/shared/ErrorMessage.svelte';
	import PathPicker from '#lib/components/shared/PathPicker.svelte';
	import Spinner from '#lib/components/shared/Spinner.svelte';
	import { jobs } from '#lib/stores/jobs.svelte.js';
	import { toasts } from '#lib/toasts.svelte.js';
	import { toRecords } from '#lib/utils/archive.js';
	import { errorMessage } from '#lib/utils/errors.js';
	import {
		CANDIDATE_JSON,
		COMPARISON_JSON,
		FAILURES,
		ROWS_CSV,
		WINDOWS_CSV,
		candidateValues,
		chainFolder,
		chainSteps,
		defaultFolder,
		lastImaging,
		lastSearch,
		newestValidation,
		resolvedProducts,
		sig,
		verdict,
		windowsArgv,
		type Verdict
	} from '#lib/utils/noise.js';

	const MIN_WINDOWS = 3;

	let busy = $state(false);
	let parameters = $state('');
	let images = $state('');
	let metadata = $state('');
	let base = $state('alma');
	let folder = $state('');

	const windowsId = $derived(page.url.searchParams.get('windows'));
	const chainIds = $derived(
		(page.url.searchParams.get('chain') ?? '').split(',').filter((id) => id.trim())
	);

	let fetched = $state<Record<string, Run>>({});
	const runOf = (id: string | null | undefined) => (id ? (jobs.get(id) ?? fetched[id]) : undefined);
	const windowsRun = $derived(runOf(windowsId));
	const chain = $derived(chainIds.map((id) => runOf(id)));

	const products = $derived(resolvedProducts(jobs.list));

	// Filled in once from your latest runs; clearing a field afterwards keeps it clear.
	let prefilled = false;
	$effect(() => {
		if (prefilled || !jobs.loaded) return;
		prefilled = true;
		const imaging = lastImaging(jobs.list);
		if (imaging && !parameters && !images) {
			parameters = imaging.parameters;
			images = imaging.images;
		}
		metadata ||= lastSearch(jobs.list) ?? '';
	});
	// The output folder follows the imaging folder until it is changed by hand.
	let folderDefault = '';
	$effect(() => {
		const next = defaultFolder(images);
		if (folder === folderDefault) folder = next;
		folderDefault = next;
	});

	function go(params: Record<string, string | null>) {
		const next = new URLSearchParams(page.url.searchParams.toString());
		for (const [key, value] of Object.entries(params)) {
			if (value === null) next.delete(key);
			else next.set(key, value);
		}
		const query = next.toString();
		// eslint-disable-next-line svelte/no-navigation-without-resolve -- the path is resolved; only the query string is appended
		void goto(`${resolve('/noise')}${query ? `?${query}` : ''}`, { replaceState: true });
	}

	$effect(() => {
		for (const id of [windowsId, ...chainIds]) {
			if (id && !jobs.get(id) && !fetched[id] && jobs.loaded) {
				void api
					.GET('/api/v2/jobs/{run_id}', { params: { path: { run_id: id } } })
					.then(({ data }) => data && (fetched = { ...fetched, [id]: data }))
					.catch(() => undefined);
			}
		}
	});

	async function readCsv(path: string) {
		const { data } = await api.GET('/api/v2/files/csv', {
			params: { query: { path, start: 0, rows: 5000 } }
		});
		return toRecords(data?.columns ?? [], data?.rows ?? []);
	}

	async function readJson(path: string): Promise<unknown> {
		const { data } = await api.GET('/api/v2/files/json', { params: { query: { path } } });
		return data ?? null;
	}

	// 2. The windows the listing found, and those it left out.
	let windows = $state<Record<string, string>[] | null>(null);
	let sources = $state<Record<string, string>>({});
	let leftOut = $state<string[]>([]);
	let windowsError = $state<unknown>(null);
	let chosen = $state(new Set<string>());
	$effect(() => {
		const run = windowsRun;
		windows = null;
		windowsError = null;
		leftOut = [];
		if (run?.state !== 'succeeded') return;
		let stale = false;
		void (async () => {
			try {
				const [table, rows] = await Promise.all([
					readCsv(`${run.run_dir}/${WINDOWS_CSV}`),
					readCsv(`${run.run_dir}/${ROWS_CSV}`)
				]);
				const log = await api.GET('/api/v2/files/text', {
					params: { query: { path: `${run.run_dir}/stdout.log` } }
				});
				if (stale) return;
				windows = table;
				sources = Object.fromEntries(
					rows.map((r) => [r.name, `${r.ALMA_source_name} · band ${r.Band} · ${r.Array_type}`])
				);
				leftOut = (log.data?.text ?? '')
					.split('\n')
					.filter((line) => line.trim().startsWith('left out '))
					.map((line) => line.trim().slice('left out '.length));
				chosen = new Set(table.map((w) => w.name));
			} catch (err) {
				if (!stale) windowsError = err;
			}
		})();
		return () => (stale = true);
	});

	// 4. What the chain made: each step's failed windows, the candidate, the verdict.
	const folderOfChain = $derived(chain[0] ? chainFolder(chain[0].argv) : null);
	let failures = $state<Record<string, Record<string, string>>>({});
	let candidate = $state<(ReturnType<typeof candidateValues> & { label: string }) | null>(null);
	let result = $state<Verdict | null>(null);
	const stepKeys = ['windows', 'extract', 'fit', 'aggregate', 'validate'] as const;
	const titles: Record<string, string> = {
		windows: 'Table the windows',
		extract: "Measure the windows' noise",
		fit: 'Fit their calibration residuals',
		aggregate: 'Reduce into a candidate model',
		validate: 'Validate the candidate'
	};
	const finishedKey = $derived(chain.map((r) => `${r?.id}:${r?.state}`).join(','));
	$effect(() => {
		void finishedKey;
		const f = folderOfChain;
		if (!f) return;
		let stale = false;
		void (async () => {
			const next: Record<string, Record<string, string>> = {};
			for (const [i, key] of stepKeys.entries()) {
				if (!FAILURES[key] || !chain[i]?.finished) continue;
				try {
					next[key] = ((await readJson(`${f}/${FAILURES[key]}`)) ?? {}) as Record<string, string>;
				} catch {
					// no failures file: the step did not get that far
				}
			}
			let found: typeof candidate = null;
			if (chain[3]?.state === 'succeeded') {
				try {
					const model = await readJson(`${f}/${CANDIDATE_JSON}`);
					const m = model as { name?: string; version?: number } | null;
					found = { ...candidateValues(model), label: `${m?.name ?? ''}@${m?.version ?? ''}` };
				} catch {
					found = null;
				}
			}
			let judged: Verdict | null = null;
			if (chain[4]?.finished) {
				try {
					const { data } = await api.GET('/api/v2/files/list', {
						params: { query: { path: `${f}/validation` } }
					});
					const newest = newestValidation(data?.entries ?? []);
					if (newest)
						judged = verdict(await readJson(`${f}/validation/${newest}/${COMPARISON_JSON}`));
				} catch {
					judged = null;
				}
			}
			if (stale) return;
			failures = next;
			candidate = found;
			result = judged;
		})();
		return () => (stale = true);
	});

	async function submit(body: { argv: string[]; title: string; after?: string[] }) {
		const { data } = await api.POST('/api/v2/jobs', { body });
		if (data) jobs.put(data);
		return data ?? null;
	}

	const source = $derived({
		parameters: parameters.trim(),
		images: images.trim(),
		metadata: metadata.trim(),
		products
	});

	async function listWindows(event: SubmitEvent) {
		event.preventDefault();
		busy = true;
		try {
			const run = await submit({
				argv: windowsArgv(source),
				title: `Noise windows · ${source.images.split('/').pop()}`
			});
			if (run) go({ windows: run.id, chain: null });
		} catch (err) {
			toasts.show(`Could not start it: ${errorMessage(err)}`, 'danger');
		} finally {
			busy = false;
		}
	}

	async function build() {
		if (!windows) return;
		busy = true;
		try {
			const names = windows.map((w) => w.name).filter((n) => chosen.has(n));
			const ids: string[] = [];
			for (const step of chainSteps(source, names, folder.trim(), base)) {
				const run = await submit({
					argv: step.argv,
					title: `Noise · ${step.title.toLowerCase()}`,
					after: ids.length ? [ids[ids.length - 1]] : []
				});
				if (!run) throw new Error(`${step.title} was not accepted`);
				ids.push(run.id);
			}
			go({ chain: ids.join(',') });
			toasts.show('The chain started; each step runs when the one before succeeds.', 'ok');
		} catch (err) {
			toasts.show(`Could not start the chain: ${errorMessage(err)}`, 'danger');
		} finally {
			busy = false;
		}
	}

	function toggle(name: string) {
		const next = new Set(chosen);
		if (next.has(name)) next.delete(name);
		else next.add(name);
		chosen = next;
	}

	const step = (n: number, done: boolean) =>
		`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-sm font-bold text-white ${done ? 'bg-ok' : 'bg-navy'}`;
</script>

<svelte:head><title>Noise models · ALMASim</title></svelte:head>

<div class="flex flex-col gap-5">
	<div class="flex flex-col gap-1">
		<p class="alma-kicker">Workflow</p>
		<h1 class="m-0 text-3xl">Noise models</h1>
		<p class="m-0 text-fg-muted">
			Learn a noise model from real data: measure the noise of the windows an imaging run made,
			reduce it into a candidate model, and check the candidate by simulating those windows again.
			Every step is a Slurm job.
		</p>
	</div>

	<!-- 1. Imaging run -->
	<section class="alma-panel flex flex-col gap-4 rounded-lg p-5" aria-labelledby="n1">
		<div class="flex flex-wrap items-center gap-3">
			<span class={step(1, windowsRun?.state === 'succeeded')}>1</span>
			<h2 id="n1" class="m-0 text-xl font-semibold">Imaging run</h2>
			{#if windowsRun}
				<StateChip state={windowsRun.state} />
				<a class="text-sm" href={resolve('/jobs/[id]', { id: windowsRun.id })}
					>job {windowsRun.short_id}</a
				>
				<button
					type="button"
					class="btn-ghost ml-auto px-3 py-1.5"
					onclick={() => go({ windows: null, chain: null })}>Start over</button
				>
			{/if}
		</div>
		{#if !windowsId}
			<form class="flex flex-col gap-4" onsubmit={listWindows}>
				<PathPicker
					bind:value={parameters}
					label="Imaging parameters CSV"
					kind="file"
					required
					hint="The windows to learn from: those the imaging run finished, science targets only."
				/>
				<PathPicker
					bind:value={images}
					label="Imaging output folder"
					required
					hint="Each window's task: its clean model and log, which the fit and the line channels need."
				/>
				<PathPicker
					bind:value={metadata}
					label="Archive search CSV"
					kind="file"
					required
					hint="The observations' archive rows: their PWV, and what the validation simulates."
				/>
				<p class="m-0 text-xs text-fg-muted">
					{#if products.length}
						Each MS is linked to its observation through the products of your {products.length}
						resolved search{products.length === 1 ? '' : 'es'}.
					{:else}
						No resolved products yet: the archive page's step 2 makes them, and they link each MS to
						its observation.
					{/if}
				</p>
				<div>
					<button
						type="submit"
						class="btn-primary"
						disabled={busy || !source.parameters || !source.images || !source.metadata}
						>List the windows</button
					>
				</div>
			</form>
		{:else if windowsRun && !windowsRun.finished}
			<p class="m-0 flex items-center gap-2 text-sm text-fg-muted">
				<Spinner /> Listing the windows…
			</p>
		{:else if windowsRun && windowsRun.state !== 'succeeded'}
			<ErrorMessage
				error={`noise windows ${windowsRun.state}: ${windowsRun.reason || 'see its log.'}`}
				title="No windows"
			/>
		{/if}
	</section>

	<!-- 2. Windows -->
	{#if windowsRun?.state === 'succeeded'}
		<section class="alma-panel flex flex-col gap-4 rounded-lg p-5" aria-labelledby="n2">
			<div class="flex flex-wrap items-center gap-3">
				<span class={step(2, chainIds.length > 0)}>2</span>
				<h2 id="n2" class="m-0 text-xl font-semibold">Windows</h2>
				{#if windows}
					<span class="text-sm text-fg-muted">{chosen.size} of {windows.length} ticked</span>
				{/if}
			</div>
			{#if windowsError}
				<ErrorMessage error={windowsError} title="Could not read the windows" />
			{:else if !windows}
				<p class="flex items-center gap-2 text-sm text-fg-muted"><Spinner /> Reading them…</p>
			{:else}
				<div class="overflow-x-auto">
					<table class="w-full text-sm">
						<thead>
							<tr class="text-left text-xs font-semibold tracking-wide text-fg-muted uppercase">
								<th scope="col" class="px-3 py-2"><span class="sr-only">Use</span></th>
								<th scope="col" class="px-3 py-2">Window</th>
								<th scope="col" class="px-3 py-2">Observation</th>
								<th scope="col" class="px-3 py-2 text-right">PWV mm</th>
							</tr>
						</thead>
						<tbody>
							{#each windows as w (w.name)}
								<tr class="border-t border-sunk">
									<td class="px-3 py-1.5">
										<input
											type="checkbox"
											aria-label="Use {w.name}"
											checked={chosen.has(w.name)}
											disabled={chainIds.length > 0}
											onchange={() => toggle(w.name)}
										/>
									</td>
									<td class="px-3 py-1.5 font-mono text-xs">{w.name}</td>
									<td class="max-w-72 truncate px-3 py-1.5" title={sources[w.name]}
										>{sources[w.name] ?? ''}</td
									>
									<td class="px-3 py-1.5 text-right font-mono">{w.pwv || '—'}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
				{#if leftOut.length}
					<details class="text-sm">
						<summary class="cursor-pointer text-fg-muted">{leftOut.length} left out</summary>
						<ul class="mt-2 font-mono text-xs">
							{#each leftOut as line (line)}<li>{line}</li>{/each}
						</ul>
					</details>
				{/if}
			{/if}
		</section>
	{/if}

	<!-- 3. Build -->
	{#if windows?.length && !chainIds.length}
		<section class="alma-panel flex flex-col gap-4 rounded-lg p-5" aria-labelledby="n3">
			<div class="flex items-center gap-3">
				<span class={step(3, false)}>3</span>
				<h2 id="n3" class="m-0 text-xl font-semibold">Build a candidate</h2>
			</div>
			<label class="flex max-w-xs flex-col gap-1 text-sm">
				<span class="font-medium">Start from</span>
				<input class="rounded-md border px-3 py-2 font-mono" bind:value={base} />
				<span class="text-xs text-fg-dim"
					>A released model (<code>alma</code>, <code>alma@1</code>) or a JSON file: it keeps the
					values too few windows measure.</span
				>
			</label>
			<PathPicker
				bind:value={folder}
				label="Output folder"
				allowCreate
				required
				hint="The measurements, candidate.json and validation/ go here."
			/>
			{#if chosen.size < MIN_WINDOWS}
				<p class="m-0 text-sm text-warn">
					With fewer than {MIN_WINDOWS} windows every value keeps the base model's.
				</p>
			{/if}
			<div>
				<button
					type="button"
					class="btn-accent"
					disabled={busy || chosen.size === 0 || !folder.trim()}
					onclick={build}
					>Measure {chosen.size} window{chosen.size === 1 ? '' : 's'} and build a candidate</button
				>
			</div>
		</section>
	{/if}

	<!-- 4. Progress and results -->
	{#if chainIds.length}
		<section class="alma-panel flex flex-col gap-4 rounded-lg p-5" aria-labelledby="n4">
			<div class="flex flex-wrap items-center gap-3">
				<span class={step(3, chain[4]?.state === 'succeeded')}>3</span>
				<h2 id="n4" class="m-0 text-xl font-semibold">The chain</h2>
				{#if folderOfChain}<code class="text-xs">{folderOfChain}</code>{/if}
			</div>
			<ol class="m-0 flex list-none flex-col gap-2 p-0">
				{#each stepKeys as key, i (key)}
					{@const run = chain[i]}
					<li class="flex flex-wrap items-center gap-3 border-t border-sunk pt-2 text-sm">
						<span class="w-64 font-medium">{titles[key]}</span>
						{#if run}
							<StateChip state={run.state} />
							<a href={resolve('/jobs/[id]', { id: run.id })}>job {run.short_id}</a>
							{#if run.state === 'pending'}
								<span class="text-fg-muted">waits for the step before</span>
							{/if}
							{#if failures[key] && Object.keys(failures[key]).length}
								<details>
									<summary class="cursor-pointer text-danger"
										>{Object.keys(failures[key]).length} window(s) failed</summary
									>
									<ul class="mt-1 font-mono text-xs">
										{#each Object.entries(failures[key]) as [name, error] (name)}
											<li><strong>{name}</strong>: {error.split('\n').pop()}</li>
										{/each}
									</ul>
								</details>
							{/if}
							{#if key === 'extract' || key === 'fit' || key === 'validate'}
								<span class="ml-auto"
									><DashboardButton runId={run.id} finished={run.finished} /></span
								>
							{/if}
						{/if}
					</li>
				{/each}
			</ol>
		</section>

		{#if candidate}
			<section class="alma-panel flex flex-col gap-4 rounded-lg p-5" aria-labelledby="n5">
				<div class="flex flex-wrap items-center gap-3">
					<h2 id="n5" class="m-0 text-xl font-semibold">Candidate {candidate.label}</h2>
					{#if folderOfChain}
						<!-- eslint-disable svelte/no-navigation-without-resolve -- the path is resolved; only the query string is appended -->
						<a
							class="btn-secondary ml-auto px-3 py-1.5 no-underline"
							href={`${resolve('/simulate')}?noise=${encodeURIComponent(`${folderOfChain}/${CANDIDATE_JSON}`)}`}
							>Use in Simulate</a
						>
						<!-- eslint-enable svelte/no-navigation-without-resolve -->
					{/if}
				</div>
				<p class="m-0 text-sm text-fg-muted">
					Each value the windows measure is their median; the range is their 16–84%.
				</p>
				<div class="overflow-x-auto">
					<table class="text-sm">
						<thead>
							<tr class="text-left text-xs font-semibold tracking-wide text-fg-muted uppercase">
								<th scope="col" class="px-3 py-2">Value</th>
								<th scope="col" class="px-3 py-2 text-right">Median</th>
								<th scope="col" class="px-3 py-2 text-right">16–84%</th>
								<th scope="col" class="px-3 py-2 text-right">Windows</th>
							</tr>
						</thead>
						<tbody>
							{#each candidate.values as v (v.key)}
								<tr class="border-t border-sunk">
									<td class="px-3 py-1 font-mono text-xs">{v.key}</td>
									<td class="px-3 py-1 text-right font-mono">{sig(v.median)}</td>
									<td class="px-3 py-1 text-right font-mono">{sig(v.q16)}–{sig(v.q84)}</td>
									<td class="px-3 py-1 text-right font-mono">{v.n}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
				{#if candidate.kept.length}
					<details class="text-sm">
						<summary class="cursor-pointer text-fg-muted"
							>{candidate.kept.length} value(s) kept from the base model</summary
						>
						<ul class="mt-2 font-mono text-xs">
							{#each candidate.kept as note (note)}<li>{note}</li>{/each}
						</ul>
					</details>
				{/if}
			</section>
		{/if}

		{#if result}
			<section class="alma-panel flex flex-col gap-4 rounded-lg p-5" aria-labelledby="n6">
				<div class="flex flex-wrap items-center gap-3">
					<h2 id="n6" class="m-0 text-xl font-semibold">Validation</h2>
					<span
						class="rounded px-2 py-0.5 text-sm font-semibold {result.passed
							? 'bg-ok-soft text-ok'
							: 'bg-danger text-white'}">{result.passed ? 'passed' : 'did not pass'}</span
					>
					<span class="text-sm text-fg-muted">{result.label} · {result.windows} window(s)</span>
				</div>
				<ul class="m-0 flex flex-col gap-1 p-0 text-sm">
					{#each result.checks as check (check.name)}
						<li class="flex gap-2">
							<span class={check.passed ? 'text-ok' : 'text-danger'}
								>{check.passed ? '✓' : '✗'}</span
							>
							<span><strong>{check.name}</strong> {check.detail}</span>
						</li>
					{/each}
				</ul>
				<div class="overflow-x-auto">
					<table class="text-sm">
						<thead>
							<tr class="text-left text-xs font-semibold tracking-wide text-fg-muted uppercase">
								<th scope="col" class="px-3 py-2">Metric (simulated vs real)</th>
								<th scope="col" class="px-3 py-2 text-right">Median</th>
								<th scope="col" class="px-3 py-2 text-right">16–84%</th>
								<th scope="col" class="px-3 py-2 text-right">p</th>
								<th scope="col" class="px-3 py-2 text-right">Within 10%</th>
								<th scope="col" class="px-3 py-2 text-right">Windows</th>
							</tr>
						</thead>
						<tbody>
							{#each result.metrics as m (m.metric)}
								<tr class="border-t border-sunk">
									<td class="px-3 py-1 font-mono text-xs"
										>{m.metric} <span class="text-fg-dim">({m.kind})</span></td
									>
									<td class="px-3 py-1 text-right font-mono">{sig(m.median)}</td>
									<td class="px-3 py-1 text-right font-mono">{sig(m.q16)}–{sig(m.q84)}</td>
									<td class="px-3 py-1 text-right font-mono">{sig(m.p, 2)}</td>
									<td class="px-3 py-1 text-right font-mono">{m.within10 ?? '—'}</td>
									<td class="px-3 py-1 text-right font-mono">{m.n}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
				<!-- Release (plan N5): a passing candidate goes to the site's models folder here. -->
			</section>
		{/if}
	{/if}
</div>
