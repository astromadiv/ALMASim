import { describe, expect, it } from 'vitest';

import { shownText, splitChunk, visibleWindow } from './logLines';

describe('splitChunk', () => {
	it('keeps an unfinished line for the next chunk', () => {
		const first = splitChunk('', 'one\ntwo\nthr');
		expect(first).toEqual({ lines: ['one', 'two'], partial: 'thr' });
		expect(splitChunk(first.partial, 'ee\nfour\n')).toEqual({
			lines: ['three', 'four'],
			partial: ''
		});
	});

	it('drops carriage returns and handles empty chunks', () => {
		expect(splitChunk('', 'a\r\nb\r\n')).toEqual({ lines: ['a', 'b'], partial: '' });
		expect(splitChunk('x', '')).toEqual({ lines: [], partial: 'x' });
	});

	it('keeps only the last redraw of a progress bar', () => {
		const bar = '\rDownloading:   0%| 0/9\rDownloading:  50%| 4/9\rDownloading: 100%| 9/9\n';
		expect(splitChunk('', `${bar}Using CASA data\n`)).toEqual({
			lines: ['Downloading: 100%| 9/9', 'Using CASA data'],
			partial: ''
		});
	});
});

describe('shownText', () => {
	it('shows what follows the last carriage return', () => {
		expect(shownText('a\rbb\rccc')).toBe('ccc');
		expect(shownText('ccc\r')).toBe('ccc');
		expect(shownText('\rccc\r\r')).toBe('ccc');
		expect(shownText('plain')).toBe('plain');
	});
});

describe('visibleWindow', () => {
	it('covers the view plus the overscan, inside the log', () => {
		expect(visibleWindow(0, 400, 20, 10_000, 10)).toEqual({ start: 0, end: 30 });
		expect(visibleWindow(2000, 400, 20, 10_000, 10)).toEqual({ start: 90, end: 130 });
		expect(visibleWindow(199_800, 400, 20, 10_000, 10)).toEqual({ start: 9980, end: 10_000 });
	});

	it('is empty for an empty log', () => {
		expect(visibleWindow(0, 400, 20, 0)).toEqual({ start: 0, end: 0 });
	});
});
