/**
 * The user guide's screenshots (web console plan, R5.2), taken from the same
 * stand-in console as the other end-to-end tests. Skipped unless
 * E2E_SCREENSHOTS names the folder to write them to:
 *
 *   LOGNAME=astro USER=astro E2E_SCREENSHOTS=$PWD/../docs/images/console \
 *     ALMASIM=... PYTHON=... npx playwright test e2e/screenshots.spec.ts
 *
 * (LOGNAME and USER name the runs' owner on the pictures.)
 */
import { expect, test, type Page } from '@playwright/test';

import { E2E_ROOT } from '../playwright.config';

const OUT = process.env.E2E_SCREENSHOTS;

test.skip(!OUT, 'E2E_SCREENSHOTS is not set');
test.use({ viewport: { width: 1280, height: 860 }, locale: 'en-US' });

async function settle(page: Page) {
	await expect(page.locator('header').getByText(/active jobs?/)).toBeVisible();
	await page.waitForTimeout(1500);
}

async function shoot(page: Page, name: string) {
	await page.screenshot({ path: `${OUT}/${name}.png` });
}

test('the console, page by page', async ({ page }) => {
	test.setTimeout(420_000);

	await page.goto('run?command=image%20image-from-ms');
	await settle(page);
	await shoot(page, 'run');

	await page.goto('archive');
	await page.getByRole('textbox', { name: /Project code/ }).fill('2023.1.00133.S');
	await page.getByRole('button', { name: 'Search the archive' }).click();
	await expect(page.getByText('3 of 3 rows', { exact: false })).toBeVisible();
	await page.getByRole('checkbox', { name: 'Tick row 2', exact: true }).check();
	await shoot(page, 'archive');

	await page.getByRole('link', { name: 'Simulate them instead' }).click();
	await expect(page.getByRole('heading', { name: 'Simulate observations' })).toBeVisible();
	await settle(page);
	await shoot(page, 'simulate');

	await page.goto('imaging');
	await page.getByRole('textbox', { name: /Folder of calibrated/ }).fill(`${E2E_ROOT}/data/cal`);
	await page.getByRole('button', { name: 'Work out the imaging parameters' }).click();
	await expect(page.locator('table tbody tr')).toHaveCount(2);
	await page.getByRole('button', { name: /^Image 2 windows/ }).click();
	await expect(page.getByRole('columnheader', { name: 'S/N' })).toBeVisible({ timeout: 60_000 });
	await page.waitForTimeout(1000);
	await shoot(page, 'imaging');

	await page.getByRole('list', { name: 'Windows' }).getByRole('link').first().click();
	await page.getByRole('button', { name: 'wsclean-image.fits' }).click();
	await expect(page.getByLabel(/wsclean-image\.fits, channel 1/)).toBeVisible();
	await page.getByRole('button', { name: '+ compare' }).last().click();
	await expect(page.getByLabel(/wsclean-residual\.fits, channel 1/)).toBeVisible();
	await page.waitForTimeout(1000);
	await shoot(page, 'products');

	// Filled in from the imaging run and the archive search above.
	await page.goto('noise');
	await settle(page);
	await page.getByRole('button', { name: 'List the windows' }).click();
	await expect(page.locator('table tbody tr')).toHaveCount(2, { timeout: 60_000 });
	await page.getByRole('button', { name: /^Measure 2 windows and build/ }).click();
	await expect(page.getByText('passed', { exact: true })).toBeVisible({ timeout: 150_000 });
	// The chain at the top, the candidate and its validation below it.
	await page
		.getByRole('heading', { name: 'The chain' })
		.evaluate((h) => window.scrollTo(0, h.getBoundingClientRect().top + window.scrollY - 110));
	await page.waitForTimeout(1000);
	await shoot(page, 'noise');

	await page.goto('jobs');
	await settle(page);
	await shoot(page, 'jobs');

	await page.getByRole('table').getByRole('link').first().click();
	await expect(page.getByRole('log')).toBeVisible();
	// The compute node's name is the machine the pictures were taken on.
	await page
		.locator('dt', { hasText: 'Host' })
		.locator('xpath=following-sibling::dd[1]')
		.evaluate((dd) => (dd.textContent = 'node01'));
	await page.waitForTimeout(1000);
	await shoot(page, 'job');

	await page.goto('./');
	await settle(page);
	await shoot(page, 'home');
});
