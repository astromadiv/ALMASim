<!--
	The simulation's noise model: none (white thermal noise only), one of the
	released models (ALMASim's and the team's, from /api/v2/noise-models), or
	a JSON file; and for a listed model, a tick per component it applies.
	It reads and writes the form's noise_model, noise_only and noise_without.
-->
<script lang="ts">
	import { api } from '#lib/api/v2/client.js';
	import ErrorMessage from '#lib/components/shared/ErrorMessage.svelte';
	import PathPicker from '#lib/components/shared/PathPicker.svelte';
	import type { Values } from '#lib/utils/commandForm.js';
	import {
		FILE_CHOICE,
		choiceOf,
		componentOptions,
		modelLabel,
		provenanceLine,
		tickedBy,
		type NoiseModelInfo
	} from '#lib/utils/noise.js';

	interface Props {
		values: Values;
	}

	let { values = $bindable() }: Props = $props();

	const text = (name: string) => (typeof values[name] === 'string' ? (values[name] as string) : '');
	const model = $derived(text('noise_model'));
	const only = $derived(text('noise_only'));
	const without = $derived(text('noise_without'));
	const setModel = (next: string) => (values.noise_model = next);

	let models = $state<NoiseModelInfo[] | null>(null);
	let siteFolder = $state<string | null>(null);
	let loadError = $state<unknown>(null);
	let fileChosen = $state(false);

	$effect(() => {
		void (async () => {
			try {
				const { data } = await api.GET('/api/v2/noise-models');
				models = data?.models ?? [];
				siteFolder = data?.site_folder ?? null;
			} catch (err) {
				loadError = err;
			}
		})();
	});

	const choice = $derived(
		fileChosen || (models !== null && choiceOf(model, models) === FILE_CHOICE)
			? FILE_CHOICE
			: models
				? choiceOf(model, models)
				: ''
	);
	const chosen = $derived(models?.find((m) => m.spec === choice) ?? null);
	const ticked = $derived(new Set(chosen ? tickedBy(chosen, only, without) : []));

	function pick(next: string) {
		fileChosen = next === FILE_CHOICE;
		values.noise_model = fileChosen ? '' : next;
		values.noise_only = '';
		values.noise_without = '';
	}

	function tick(name: string, on: boolean) {
		if (!chosen) return;
		const next = new Set(ticked);
		if (on) next.add(name);
		else next.delete(name);
		const options = componentOptions(chosen, next);
		values.noise_only = options.noise_only;
		values.noise_without = options.noise_without;
	}
</script>

<details class="alma-panel group rounded-lg" open>
	<summary
		class="flex cursor-pointer list-none items-center gap-2 px-5 py-3 text-base font-semibold text-navy"
	>
		<span class="inline-block transition-transform group-open:rotate-90" aria-hidden="true">›</span>
		Noise
	</summary>
	<div class="flex flex-col gap-4 border-t border-line px-5 py-4">
		{#if loadError}<ErrorMessage error={loadError} title="The noise models did not load" />{/if}
		<div class="flex flex-col gap-1 text-sm">
			<label for="noise-model" class="font-medium">Noise model</label>
			<select
				id="noise-model"
				class="max-w-xl rounded-md border px-3 py-2"
				aria-describedby="noise-model-hint"
				value={choice}
				onchange={(e) => pick(e.currentTarget.value)}
			>
				<option value="">None: white thermal noise only</option>
				{#each models ?? [] as m (m.spec)}
					<option value={m.spec}>{modelLabel(m)}</option>
				{/each}
				<option value={FILE_CHOICE}>A model file (JSON)…</option>
			</select>
			<span id="noise-model-hint" class="text-xs text-fg-dim">
				What the visibilities carry besides white thermal noise.
				{siteFolder ? `Team models come from ${siteFolder}.` : ''}
			</span>
		</div>

		{#if choice === FILE_CHOICE}
			<PathPicker bind:value={() => model, setModel} label="Model file" kind="file" required />
		{:else if chosen}
			<div class="flex flex-col gap-1 text-sm">
				{#if chosen.description}<p class="m-0">{chosen.description}</p>{/if}
				{#if provenanceLine(chosen)}
					<p class="m-0 text-xs text-fg-dim">{provenanceLine(chosen)}</p>
				{/if}
			</div>
			<fieldset class="flex flex-col gap-2 text-sm">
				<legend class="mb-1 font-medium">
					Components ({ticked.size} of {chosen.components.filter((c) => c.enabled).length} applied)
				</legend>
				{#each chosen.components as c (c.name)}
					<label class="flex items-start gap-2" class:text-fg-dim={!c.enabled}>
						<input
							type="checkbox"
							class="mt-1"
							checked={ticked.has(c.name)}
							disabled={!c.enabled}
							onchange={(e) => tick(c.name, e.currentTarget.checked)}
						/>
						<span>
							<code>{c.name}</code>
							<span class="text-fg-muted">
								{c.description}{c.enabled ? '' : ' (off in this model)'}
							</span>
						</span>
					</label>
				{/each}
			</fieldset>
		{/if}
	</div>
</details>
