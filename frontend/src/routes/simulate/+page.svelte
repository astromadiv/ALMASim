<!--
	Simulate ALMA observations: pick rows of a metadata CSV (from the Archive
	page, or any saved one), choose the sky model and options, see how big
	the cubes will be, and submit one run per row. ?metadata=<csv path>
	opens a CSV, &rows=0,3 ticks rows in it.
-->
<script lang="ts">
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { api, type Run } from '#lib/api/v2/client.js';
	import type { components } from '#lib/api/v2/schema.js';
	import ResultsTable from '#lib/components/archive/ResultsTable.svelte';
	import StateChip from '#lib/components/jobs/StateChip.svelte';
	import CommandSections from '#lib/components/run/CommandSections.svelte';
	import NoisePicker from '#lib/components/simulate/NoisePicker.svelte';
	import ErrorMessage from '#lib/components/shared/ErrorMessage.svelte';
	import PathPicker from '#lib/components/shared/PathPicker.svelte';
	import Spinner from '#lib/components/shared/Spinner.svelte';
	import { jobs } from '#lib/stores/jobs.svelte.js';
	import { toasts } from '#lib/toasts.svelte.js';
	import { toRecords } from '#lib/utils/archive.js';
	import type { Command, Values } from '#lib/utils/commandForm.js';
	import { errorMessage } from '#lib/utils/errors.js';
	import {
		NOISE_FIELDS,
		ROW_FIELDS,
		batchGb,
		cubeVolume,
		cubesGb,
		simulationArgv,
		simulationCpus,
		simulationDefaults
	} from '#lib/utils/simulate.js';
	import { inRunDir } from '#lib/utils/runDir.js';

	type Estimate = components['schemas']['Estimate'];
	type Plan = components['schemas']['Plan'];
	type CubeList = components['schemas']['CubeList'];
	type InputCube = components['schemas']['InputCube'];

	let simulation = $state<Command | null>(null);
	let values = $state<Values>({});
	let loadError = $state<unknown>(null);
	let metadataPath = $state(page.url.searchParams.get('metadata') ?? '');
	let table = $state<{ columns: string[]; records: Record<string, string>[] } | null>(null);
	let tableError = $state<unknown>(null);
	let selected = $state(new Set<number>());
	let estimate = $state<Estimate | null>(null);
	let estimateError = $state<unknown>(null);
	let estimating = $state(false);
	let submitting = $state(false);
	let submitted = $state<string[]>([]);
	// What the console would submit for the first ticked row, and the CPUs the
	// site gives a simulation job (which the worker count starts from).
	let plan = $state<Plan | null>(null);
	let siteCpus = $state<number | null>(null);
	// Input cubes (optional): a cube, or a folder of them; each matched to the
	// metadata row of its observation by the console.
	let inputPath = $state(page.url.searchParams.get('input') ?? '');
	let cubeList = $state<CubeList | null>(null);
	let cubesError = $state<unknown>(null);
	let readingCubes = $state(false);

	const rows = $derived([...selected].sort((a, b) => a - b));
	const perRow = $derived(Number(values.simulations_per_row) || 1);
	const usingCubes = $derived(cubeList !== null && inputPath.trim() !== '');
	// One run per ticked row, or with cubes, per cube of a ticked row.
	const runs = $derived<{ row: number; cube: InputCube | null }[]>(
		usingCubes && cubeList
			? cubeList.cubes
					.filter((c) => selected.has(c.row_idx))
					.map((c) => ({ row: c.row_idx, cube: c }))
			: rows.map((row) => ({ row, cube: null }))
	);
	// The page's own fields (the noise picker's too), the site's (its CASA data,
	// for --ms-export), and with cubes, what a cube sets itself.
	const hidden = $derived([
		...ROW_FIELDS,
		...NOISE_FIELDS,
		'input_path',
		'products_csv',
		'casa_data_root',
		...(usingCubes ? ['n_pix', 'n_channels', 'source_type'] : [])
	]);

	$effect(() => {
		void (async () => {
			try {
				const { data } = await api.GET('/api/v2/commands');
				simulation = data?.find((c) => c.name === 'simulation') ?? null;
				if (simulation) values = simulationDefaults(simulation);
				// ?noise=<model> (the Noise models page's candidate) chooses the noise model.
				const noise = page.url.searchParams.get('noise');
				if (noise && 'noise_model' in values) values.noise_model = noise;
			} catch (err) {
				loadError = err;
			}
		})();
	});

	// The metadata rows, once a CSV is chosen.
	$effect(() => {
		const path = metadataPath.trim();
		table = null;
		tableError = null;
		if (!path.endsWith('.csv')) return;
		let stale = false;
		api
			.GET('/api/v2/files/csv', { params: { query: { path, start: 0, rows: 5000 } } })
			.then(({ data }) => {
				if (stale || !data) return;
				table = { columns: data.columns, records: toRecords(data.columns, data.rows) };
				const wanted = (page.url.searchParams.get('rows') ?? '')
					.split(',')
					.map(Number)
					.filter((i) => Number.isInteger(i) && i >= 0 && i < data.rows.length);
				selected = new Set(wanted.length ? wanted : data.rows.length === 1 ? [0] : []);
			})
			.catch((err) => !stale && (tableError = err));
		return () => (stale = true);
	});

	// The input cubes and their rows, once both paths are set; their rows get ticked.
	$effect(() => {
		const path = inputPath.trim();
		const metadata = metadataPath.trim();
		cubeList = null;
		cubesError = null;
		if (!path || !table) return;
		const controller = new AbortController();
		const timer = setTimeout(async () => {
			readingCubes = true;
			try {
				const { data } = await api.POST('/api/v2/simulation/cubes', {
					body: { metadata_path: metadata, input_path: path },
					signal: controller.signal
				});
				cubeList = data ?? null;
				if (data) selected = new Set(data.cubes.map((c) => c.row_idx));
			} catch (err) {
				if (!controller.signal.aborted) cubesError = err;
			} finally {
				readingCubes = false;
			}
		}, 400);
		return () => {
			clearTimeout(timer);
			controller.abort();
		};
	});

	// How big the cubes will be: for the first run, a moment after a change.
	$effect(() => {
		const body = {
			metadata_path: metadataPath.trim(),
			row_idx: runs[0]?.row ?? -1,
			n_pix: Number(values.n_pix) || null,
			n_channels: Number(values.n_channels) || null,
			source_type: String(values.source_type || 'point'),
			instrument_geometry: String(values.instrument_geometry || 'legacy'),
			input_path: runs[0]?.cube?.path ?? null
		};
		if (!table || body.row_idx < 0) {
			estimate = null;
			return;
		}
		const controller = new AbortController();
		const timer = setTimeout(async () => {
			estimating = true;
			try {
				const { data } = await api.POST('/api/v2/simulation/estimate', {
					body,
					signal: controller.signal
				});
				estimate = data ?? null;
				estimateError = null;
			} catch (err) {
				if (!controller.signal.aborted) {
					estimate = null;
					estimateError = err;
				}
			} finally {
				if (!controller.signal.aborted) estimating = false;
			}
		}, 500);
		return () => {
			clearTimeout(timer);
			controller.abort();
		};
	});

	// Where and how each run goes: a moment after a change, for the first run.
	$effect(() => {
		const first = runs[0];
		if (!simulation || !table || !first) {
			plan = null;
			return;
		}
		const argv = simulationArgv(
			simulation,
			values,
			metadataPath.trim(),
			first.row,
			runs.length > 1,
			first.cube
		);
		const cpus = simulationCpus(values);
		const controller = new AbortController();
		const timer = setTimeout(async () => {
			try {
				if (siteCpus === null) {
					// The site's CPUs for a simulation: the local backend gets a worker for each.
					const { data } = await api.POST('/api/v2/commands/validate', {
						body: { argv },
						signal: controller.signal
					});
					siteCpus = data?.plan?.resources.cpus ?? 1;
					if (values.backend === 'local' && Number(values.n_workers) <= 1) {
						values.n_workers = String(siteCpus);
						return; // the change runs this again
					}
				}
				const { data } = await api.POST('/api/v2/commands/validate', {
					body: { argv, resources: { cpus } },
					signal: controller.signal
				});
				plan = data?.plan ?? null;
			} catch {
				if (!controller.signal.aborted) plan = null;
			}
		}, 500);
		return () => {
			clearTimeout(timer);
			controller.abort();
		};
	});

	async function submit() {
		if (!simulation || !runs.length) return;
		submitting = true;
		const made: Run[] = [];
		try {
			for (const { row, cube } of runs) {
				const source = table?.records[row]?.ALMA_source_name || `row ${row}`;
				const { data } = await api.POST('/api/v2/jobs', {
					body: {
						argv: simulationArgv(
							simulation,
							values,
							metadataPath.trim(),
							row,
							runs.length > 1,
							cube
						),
						resources: { cpus: simulationCpus(values) },
						params: {
							...values,
							metadata_path: metadataPath.trim(),
							row_idx: row,
							input_path: cube?.path ?? ''
						},
						title: `Simulation · ${source}${cube ? ` · ${cube.label}` : ''}`
					}
				});
				if (data) {
					jobs.put(data);
					made.push(data);
				}
			}
			toasts.show(`Submitted ${made.length} simulation${made.length === 1 ? '' : 's'}`, 'ok');
		} catch (err) {
			toasts.show(
				`Submitted ${made.length} of ${runs.length}, then: ${errorMessage(err)}`,
				'danger'
			);
		} finally {
			submitted = [...made.map((r) => r.id), ...submitted];
			submitting = false;
		}
	}

	const gb = (n: number) => (n >= 100 ? n.toFixed(0) : n >= 1 ? n.toFixed(1) : n.toFixed(3));

	/** Where a run wrote: its --output-dir, with {run_dir} filled in. */
	function outputsOf(run: Run): string {
		const at = run.argv.indexOf('--output-dir');
		return at >= 0 ? inRunDir(run.argv[at + 1], run.run_dir) : run.run_dir;
	}
</script>

<svelte:head><title>Simulate · ALMASim</title></svelte:head>

<div class="flex flex-col gap-5">
	<div class="flex flex-col gap-1">
		<p class="alma-kicker">Workflow</p>
		<h1 class="m-0 text-3xl">Simulate observations</h1>
		<p class="m-0 text-fg-muted">
			Pick observations from a metadata CSV, choose a sky model, and simulate what ALMA would see:
			one job per observation.
		</p>
	</div>

	{#if loadError}<ErrorMessage error={loadError} title="Something did not load" />{/if}

	<section class="alma-panel flex flex-col gap-4 rounded-lg p-5" aria-labelledby="m-h">
		<h2 id="m-h" class="m-0 text-xl font-semibold">Observations</h2>
		<PathPicker
			bind:value={metadataPath}
			label="Metadata CSV"
			kind="file"
			hint="The results of an archive search (Archive page), or any CSV from almasim metadata."
		/>
		{#if tableError}
			<ErrorMessage error={tableError} title="Could not read the CSV" />
		{:else if metadataPath.trim().endsWith('.csv') && !table}
			<p class="flex items-center gap-2 text-sm text-fg-muted"><Spinner /> Reading…</p>
		{:else if table}
			<ResultsTable records={table.records} columns={table.columns} bind:selected />
		{/if}
		{#if table}
			<PathPicker
				bind:value={inputPath}
				label="Sky-model cubes (optional)"
				kind="path"
				hint="A cube, or a folder of them (an imaging run's: each spectral window's model cube). Each is simulated with the row of its observation, on its own grid."
			/>
			{#if cubesError}
				<ErrorMessage error={cubesError} title="Could not match the cubes" />
			{:else if readingCubes}
				<p class="flex items-center gap-2 text-sm text-fg-muted">
					<Spinner /> Matching the cubes to the metadata…
				</p>
			{:else if cubeList}
				<p class="m-0 text-sm">
					{cubeList.cubes.length} cube{cubeList.cubes.length === 1 ? '' : 's'} of
					{new Set(cubeList.cubes.map((c) => c.row_idx)).size} observation{new Set(
						cubeList.cubes.map((c) => c.row_idx)
					).size === 1
						? ''
						: 's'} (their rows are ticked): <strong>{runs.length} will run</strong>, each on its own
					grid.
					{#if cubeList.unmatched.length}
						{cubeList.unmatched.length} match no row (their path names no observation here).
					{/if}
				</p>
				{#if cubeList.cubes.length}
					<div class="max-h-72 overflow-auto">
						<table class="w-full text-sm" aria-label="Input cubes and their observations">
							<thead>
								<tr class="text-left text-xs font-semibold tracking-wide text-fg-muted uppercase">
									<th scope="col" class="px-3 py-2">Cube</th>
									<th scope="col" class="px-3 py-2">Observation</th>
									<th scope="col" class="px-3 py-2 text-right">Pixels × channels</th>
									<th scope="col" class="px-3 py-2 text-right">Centre GHz</th>
									<th scope="col" class="px-3 py-2 text-right">Channel MHz</th>
									<th scope="col" class="px-3 py-2 text-right">Pixel ″</th>
								</tr>
							</thead>
							<tbody>
								{#each cubeList.cubes as cube (`${cube.path}#${cube.row_idx}`)}
									<tr class="border-t border-sunk {selected.has(cube.row_idx) ? '' : 'opacity-50'}">
										<td class="max-w-72 truncate px-3 py-1.5 font-mono text-xs" title={cube.path}
											>{cube.label}</td
										>
										<td class="px-3 py-1.5">{cube.source_name} (row {cube.row_idx})</td>
										<td class="px-3 py-1.5 text-right font-mono"
											>{cube.n_pix_x} × {cube.n_pix_y} × {cube.n_channels ?? 1}</td
										>
										<td class="px-3 py-1.5 text-right font-mono"
											>{cube.central_freq_ghz?.toFixed(3) ?? '—'}</td
										>
										<td class="px-3 py-1.5 text-right font-mono"
											>{cube.channel_width_mhz?.toFixed(3) ?? '—'}</td
										>
										<td class="px-3 py-1.5 text-right font-mono"
											>{cube.cell_arcsec?.toFixed(3) ?? '—'}</td
										>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{/if}
			{/if}
		{/if}
	</section>

	{#if simulation && table}
		<p class="m-0 text-sm text-fg-muted">
			<code>{'{run_dir}'}</code> in a path is each run's own folder (its job page lists what is in
			it). By default every simulation keeps its FITS products and its ML shard there; a folder you
			choose instead gets one folder per run (<code>row-&lt;n&gt;</code>, or the cube's name).
		</p>
		<NoisePicker bind:values />
		<CommandSections command={simulation} bind:values hide={hidden} />

		<section class="alma-panel flex flex-col gap-3 rounded-lg p-5" aria-labelledby="e-h">
			<div class="flex items-center gap-2">
				<h2 id="e-h" class="m-0 text-xl font-semibold">Size</h2>
				{#if estimating}<Spinner />{/if}
			</div>
			{#if !runs.length}
				<p class="m-0 text-sm text-fg-muted">Tick at least one observation.</p>
			{:else if estimateError}
				<ErrorMessage error={estimateError} title="Could not estimate it" />
			{:else if estimate}
				<p class="m-0 text-sm">
					Each simulation of <strong>{estimate.source_name}</strong> (band {estimate.band}) makes a
					cube of {estimate.cube_shape.join(' × ')} (channels × pixels²), {estimate.cell_size_arcsec.toFixed(
						3
					)}″ pixels for a {estimate.beam_size_arcsec.toFixed(2)}″ beam: about
					<strong>{gb(estimate.estimated_standard_output_gb)} GB</strong> of standard outputs.
				</p>
				<p class="m-0 text-sm">
					{#if usingCubes}
						{runs.length} cube{runs.length === 1 ? '' : 's'} × {perRow} each: about
						<strong
							>{gb(
								cubesGb(
									estimate.estimated_standard_output_gb,
									runs.map((r) => (r.cube ? cubeVolume(r.cube) : 0)),
									perRow
								)
							)} GB</strong
						>
						in all (each cube's share by its size).
					{:else}
						{rows.length} observation{rows.length === 1 ? '' : 's'} × {perRow} per observation: about
						<strong
							>{gb(batchGb(estimate.estimated_standard_output_gb, rows.length, perRow))} GB</strong
						>
						in all (if the others are like the first).
					{/if}
				</p>
				<p class="m-0 text-xs text-fg-dim">{estimate.note}</p>
			{/if}
			{#if plan}
				<p class="m-0 text-sm">
					{#if plan.executor === 'slurm'}
						Each is a Slurm job: <strong
							>{plan.resources.cpus} CPU{plan.resources.cpus === 1 ? '' : 's'}</strong
						>, {plan.resources.memory}, up to {plan.resources.time} on {plan.resources.partition ??
							'the default partition'}.
					{:else}
						Each runs on the console's own machine (this site runs simulations there, not on Slurm),
						with {plan.resources.cpus} CPU{plan.resources.cpus === 1 ? '' : 's'}.
					{/if}
					{values.backend === 'local'
						? `Its channels are imaged in parallel, one worker per CPU (Cluster: --backend local, --n-workers).`
						: `Its channels are imaged one after another (Cluster: --backend sync), so it asks for one CPU.`}
				</p>
			{/if}
			<div>
				<button
					type="button"
					class="btn-accent"
					disabled={!runs.length || submitting}
					onclick={submit}
					>{submitting
						? 'Submitting…'
						: usingCubes
							? `Simulate ${runs.length} cube${runs.length === 1 ? '' : 's'}`
							: `Simulate ${rows.length} observation${rows.length === 1 ? '' : 's'}`}</button
				>
			</div>
		</section>
	{/if}

	{#if submitted.length}
		<section class="alma-panel flex flex-col gap-2 rounded-lg p-5" aria-labelledby="r-h">
			<h2 id="r-h" class="m-0 text-xl font-semibold">Submitted</h2>
			<ul class="m-0 flex list-none flex-col gap-1.5 p-0">
				{#each submitted as id (id)}
					{@const run = jobs.get(id)}
					<li class="flex items-center gap-3 text-sm">
						{#if run}<StateChip state={run.state} />{/if}
						<a href={resolve('/jobs/[id]', { id })}>{run?.title ?? id}</a>
						{#if run?.state === 'succeeded'}
							<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- the path is resolved; only the query string is appended -->
							<a href={`${resolve('/products')}?dir=${encodeURIComponent(outputsOf(run))}`}
								>Look at the results</a
							>
						{/if}
					</li>
				{/each}
			</ul>
		</section>
	{/if}
</div>
