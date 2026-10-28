<!--
	A run's log, followed live: chunks by byte offset (`/api/v2/jobs/{id}/log`),
	read again every few seconds until the run has finished and the end is
	reached. Only the lines in view are drawn, so long logs stay fast. Scrolling
	up pauses following; "Follow" jumps back to the end.
-->
<script lang="ts">
	import { tick } from 'svelte';
	import { api } from '#lib/api/v2/client.js';
	import { consoleUrl } from '#lib/config.js';
	import { shownText, splitChunk, visibleWindow } from '#lib/utils/logLines.js';
	import ErrorMessage from '#lib/components/shared/ErrorMessage.svelte';

	type Stream = 'stdout' | 'stderr';

	interface Props {
		runId: string;
		runDir: string;
		/** The run has ended: read to the end once more, then stop. */
		finished: boolean;
		/** Pause between reads while the run goes on, in milliseconds. */
		pollMs?: number;
	}

	let { runId, runDir, finished, pollMs = 2000 }: Props = $props();

	const LINE_HEIGHT = 20;
	const CHUNK = 256 * 1024;

	let stream = $state<Stream>('stdout');
	let lines = $state.raw<string[]>([]);
	let partial = $state('');
	let size = $state(0);
	let error = $state<unknown>(null);
	let follow = $state(true);
	let scrollTop = $state(0);
	let viewHeight = $state(400);
	let view: HTMLDivElement | undefined = $state();

	const total = $derived(lines.length + (partial ? 1 : 0));
	const shown = $derived(visibleWindow(scrollTop, viewHeight, LINE_HEIGHT, total));
	const indices = $derived(
		Array.from({ length: shown.end - shown.start }, (_, k) => shown.start + k)
	);
	const lineAt = (i: number) => (i < lines.length ? lines[i] : shownText(partial));

	// Read the chosen stream from the start, then keep reading while the run lasts.
	$effect(() => {
		const id = runId;
		const which = stream;
		let stopped = false;
		let timer: ReturnType<typeof setTimeout> | undefined;
		let offset = 0;
		let pending = '';
		lines = [];
		partial = '';
		size = 0;
		error = null;

		async function read() {
			try {
				const { data } = await api.GET('/api/v2/jobs/{run_id}/log', {
					params: { path: { run_id: id }, query: { stream: which, offset, limit: CHUNK } }
				});
				if (stopped || !data) return;
				const split = splitChunk(pending, data.text);
				pending = split.partial;
				if (split.lines.length) lines = [...lines, ...split.lines];
				partial = pending;
				offset = data.next_offset;
				size = data.size;
				error = null;
				if (follow) await scrollToEnd();
				if (offset < data.size) return read(); // more is already there
			} catch (err) {
				if (stopped) return;
				error = err;
			}
			if (stopped) return;
			// While the run goes on, read again later. Once it has ended, read one
			// last time (it may have written more just before it said so), then stop.
			if (!finished) timer = setTimeout(read, pollMs);
			else if (!lastRead) {
				lastRead = true;
				timer = setTimeout(read, 500);
			}
		}

		let lastRead = false;
		void read();
		return () => {
			stopped = true;
			clearTimeout(timer);
		};
	});

	async function scrollToEnd() {
		await tick();
		if (view) view.scrollTop = view.scrollHeight;
	}

	function onScroll() {
		if (!view) return;
		scrollTop = view.scrollTop;
		const atEnd = view.scrollTop + view.clientHeight >= view.scrollHeight - LINE_HEIGHT;
		if (!atEnd && follow) follow = false;
		if (atEnd && !follow) follow = true;
	}

	function resumeFollowing() {
		follow = true;
		void scrollToEnd();
	}

	const downloadHref = $derived(
		`${consoleUrl()}/api/v2/files/download?path=${encodeURIComponent(`${runDir}/${stream}.log`)}`
	);
</script>

<div class="flex flex-col">
	<div class="flex flex-wrap items-center gap-2 px-5 py-2.5">
		<div role="group" aria-label="Log stream" class="flex gap-1 rounded-full bg-sunk p-1">
			{#each ['stdout', 'stderr'] as const as option (option)}
				<button
					type="button"
					class="rounded-full px-3 py-1 font-mono text-xs text-fg-muted aria-pressed:bg-navy aria-pressed:text-white"
					aria-pressed={stream === option}
					onclick={() => (stream = option)}>{option}.log</button
				>
			{/each}
		</div>
		<span class="text-xs text-fg-muted">
			{total.toLocaleString()} lines · {(size / 1024).toFixed(size < 10240 ? 1 : 0)} kB
		</span>
		<span class="flex-1"></span>
		{#if !finished}
			<button
				type="button"
				class="btn-secondary px-3 py-1 text-xs"
				aria-pressed={follow}
				onclick={() => (follow ? (follow = false) : resumeFollowing())}
				>{follow ? 'Pause' : 'Follow'}</button
			>
		{/if}
		<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- an API download, not a page -->
		<a class="btn-secondary px-3 py-1 text-xs no-underline" href={downloadHref} download>Download</a
		>
	</div>
	{#if error}
		<div class="px-5 pb-3"><ErrorMessage {error} title="Could not read the log" /></div>
	{/if}
	<!-- svelte-ignore a11y_no_noninteractive_tabindex (a scrolling region must take focus, so the keyboard can scroll it) -->
	<div
		bind:this={view}
		bind:clientHeight={viewHeight}
		onscroll={onScroll}
		class="relative h-[min(60vh,28rem)] overflow-auto bg-fg font-mono text-xs leading-5 text-navy-soft"
		role="log"
		aria-label="{stream}.log"
		tabindex="0"
	>
		{#if total === 0}
			<p class="p-4 text-fg-dim">{finished ? 'Nothing was written here.' : 'Nothing yet.'}</p>
		{:else}
			<div style="height: {total * LINE_HEIGHT + 24}px" class="relative">
				{#each indices as i (i)}
					<div
						class="absolute right-0 left-0 px-4 whitespace-pre"
						style="top: {12 + i * LINE_HEIGHT}px; height: {LINE_HEIGHT}px"
					>
						{lineAt(i)}
					</div>
				{/each}
			</div>
		{/if}
	</div>
</div>
