/**
 * The console's workflows end to end, against a real `almasim serve` whose
 * commands are instant stand-ins (e2e/fake_almasim.py).
 */
import { mkdirSync, writeFileSync } from 'node:fs';

import { expect, test, type Page } from '@playwright/test';

import { E2E_ROOT } from '../playwright.config';

async function chip(page: Page, state: string) {
	await expect(page.getByText(state).first()).toBeVisible({ timeout: 60_000 });
}

test('run a command, follow its log, run it again', async ({ page }) => {
	await page.goto('run');
	await page
		.getByRole('navigation', { name: 'Commands' })
		.getByRole('button', { name: 'doctor' })
		.click();
	await page.getByRole('textbox', { name: /Title/ }).fill('e2e doctor');
	await expect(page.getByText('Runs on the console server.')).toBeVisible();
	await page.getByRole('button', { name: 'Run', exact: true }).click();

	await expect(page).toHaveURL(/\/jobs\/[0-9A-Z]{26}$/);
	const first = page.url();
	await chip(page, 'Succeeded');
	await expect(page.getByRole('log').getByText('fake almasim doctor')).toBeVisible();

	await page.getByRole('button', { name: 'Run again' }).click();
	await expect(page).not.toHaveURL(first);
	await chip(page, 'Succeeded');

	await page.goto('jobs');
	await expect(page.getByRole('link', { name: 'e2e doctor' })).toHaveCount(2);
});

test('cancel a long run from its page', async ({ page }) => {
	await page.goto('run?command=doctor');
	await page.getByRole('checkbox', { name: /Slurm/ }).check();
	await page.getByRole('textbox', { name: /Title/ }).fill('e2e long');
	await expect(page.getByText('Runs on the console server.')).toBeVisible();
	await page.getByRole('button', { name: 'Run', exact: true }).click();
	await chip(page, 'Running');
	await expect(
		page
			.getByRole('log')
			.getByText(/waiting for the compute node/)
			.first()
	).toBeVisible();

	await page.getByRole('button', { name: 'Cancel job' }).click();
	const dialog = page.getByRole('dialog', { name: 'Cancel this job?' });
	await dialog.getByRole('button', { name: 'Cancel job' }).click();
	await chip(page, 'Cancelled');

	// A finished run of yours can go.
	await page.getByRole('button', { name: 'Delete' }).click();
	await page
		.getByRole('dialog', { name: 'Delete this run?' })
		.getByRole('button', { name: 'Delete' })
		.click();
	await expect(page).toHaveURL(/\/jobs$/);
	await expect(page.getByRole('link', { name: 'e2e long' })).toHaveCount(0);
});

test('search the archive, resolve the products and download them', async ({ page }) => {
	await page.goto('archive');
	await page.getByRole('textbox', { name: /Project code/ }).fill('2023.1.00133.S');
	await page.getByRole('button', { name: 'Search the archive' }).click();

	await expect(page.getByText('3 of 3 rows', { exact: false })).toBeVisible();
	await page.getByRole('checkbox', { name: 'Tick row 2', exact: true }).check();
	await page.getByRole('button', { name: /Find the products of 1 observation/ }).click();

	const download = page.getByRole('combobox', { name: 'Download' });
	await expect(download).toBeVisible();
	await expect(page.getByText('953.7 MB').first()).toBeVisible(); // the raw product (10⁹ bytes)
	const only = page.getByRole('textbox', { name: 'Only files named' });
	await only.fill('*_cubes.tar');
	await expect(page.getByText(/^1 file, 190\.7 MB to download/)).toBeVisible();
	await only.fill('');
	await download.selectOption({ label: 'auxiliary only' });
	await page.getByRole('textbox', { name: /Destination/ }).fill(`${E2E_ROOT}/data/archive`);
	await expect(page.getByText(/free there/)).toBeVisible();
	await page.getByRole('button', { name: /^Download / }).click();

	await expect(page).toHaveURL(/\/jobs\/[0-9A-Z]{26}$/);
	await chip(page, 'Succeeded');
});

test('image a folder, then look at the results', async ({ page }) => {
	await page.goto('imaging');
	await page.getByRole('textbox', { name: /Folder of calibrated/ }).fill(`${E2E_ROOT}/data/cal`);
	await page.getByRole('button', { name: 'Work out the imaging parameters' }).click();

	await expect(page.locator('table tbody tr')).toHaveCount(2);
	await expect(page.getByRole('textbox', { name: /Output folder/ })).toHaveValue(
		`${E2E_ROOT}/data/cal-images`
	);
	await page.getByRole('button', { name: /^Image 2 windows/ }).click();

	await expect(page.getByText(/^2 done · 0 running · 0 failed · 0 waiting/)).toBeVisible({
		timeout: 60_000
	});
	await expect(page.getByRole('columnheader', { name: 'S/N' })).toBeVisible({ timeout: 60_000 });

	await page.getByRole('list', { name: 'Windows' }).getByRole('link').first().click();
	await expect(page.getByRole('heading', { name: 'How well it was cleaned' })).toBeVisible();
	await page.getByRole('button', { name: 'wsclean-image.fits' }).click();
	await expect(page.getByLabel(/wsclean-image\.fits, channel 1/)).toBeVisible();
	await expect(page.getByText(/beam 2\.50″ × 2\.00″/)).toBeVisible();
	await page.getByRole('button', { name: '+ compare' }).last().click();
	await expect(page.getByLabel(/wsclean-residual\.fits, channel 1/)).toBeVisible();

	// A preview computed as a job opens beside them when it is done.
	await page.getByRole('button', { name: 'map' }).first().click();
	await expect(page.getByText('opened below')).toBeVisible({ timeout: 60_000 });
	await expect(page.getByLabel(/-map\.fits, channel 1/)).toBeVisible();
});

test('build a noise model from an imaging run', async ({ page }) => {
	test.setTimeout(180_000); // five jobs, one after the other
	// The inputs need only exist: the fake noise commands make up their windows.
	const inputs = `${E2E_ROOT}/data/noise-inputs`;
	mkdirSync(`${inputs}/images`, { recursive: true });
	writeFileSync(`${inputs}/params.csv`, 'filename,spectral_window_id\n');
	writeFileSync(`${inputs}/results.csv`, 'member_ous_uid\n');

	await page.goto('noise');
	await page.getByRole('textbox', { name: /Imaging parameters CSV/ }).fill(`${inputs}/params.csv`);
	await page.getByRole('textbox', { name: /Imaging output folder/ }).fill(`${inputs}/images`);
	await page.getByRole('textbox', { name: /Archive search CSV/ }).fill(`${inputs}/results.csv`);
	await page.getByRole('button', { name: 'List the windows' }).click();

	await expect(page.locator('table tbody tr')).toHaveCount(2, { timeout: 60_000 });
	await expect(page.getByText('1 left out')).toBeVisible();
	await page.getByRole('checkbox', { name: /spw27/ }).uncheck();
	await expect(page.getByText('1 of 2 ticked')).toBeVisible();
	await expect(page.getByRole('textbox', { name: /Output folder/ })).toHaveValue(
		`${inputs}/images-noise`
	);
	await page.getByRole('button', { name: /^Measure 1 window and build/ }).click();

	await expect(page.getByRole('heading', { name: 'Candidate alma-candidate@2' })).toBeVisible({
		timeout: 90_000
	});
	await expect(page.getByRole('cell', { name: 'channel_correlation.rho1' })).toBeVisible();
	await expect(page.getByText('1 window(s) failed')).toBeVisible();
	await expect(page.getByText('passed', { exact: true })).toBeVisible({ timeout: 60_000 });
	await expect(page.getByRole('cell', { name: /cont\.sigma/ })).toBeVisible();

	// Release it as a team model: never under a package model's name.
	const releaseName = page.getByRole('combobox', { name: /^Model name/ });
	await releaseName.fill('alma');
	await expect(page.getByText('alma is a model ALMASim releases')).toBeVisible();
	await expect(page.getByRole('button', { name: 'Release…' })).toBeDisabled();
	await releaseName.fill('e2e-alma');
	await page.getByRole('button', { name: 'Release…' }).click();
	await page.getByRole('dialog').getByRole('button', { name: 'Release', exact: true }).click();
	await expect(page.getByText('Released as e2e-alma@1')).toBeVisible({ timeout: 60_000 });
	await expect(page.getByRole('link', { name: 'Use e2e-alma@1 in Simulate' })).toHaveAttribute(
		'href',
		/simulate\?noise=e2e-alma%401$/
	);

	await page.getByRole('link', { name: 'Use in Simulate' }).click();
	await expect(page).toHaveURL(/simulate\?noise=/);
});
