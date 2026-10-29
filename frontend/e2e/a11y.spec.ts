/** Every page, checked with axe: no serious or critical WCAG 2 A/AA problems. */
import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

const PAGES = [
	'/',
	'/run',
	'/run?command=image%20image-from-ms',
	'/jobs',
	'/archive',
	'/simulate',
	'/imaging',
	'/noise',
	'/products',
	'/settings'
];

for (const path of PAGES) {
	test(`accessible: ${path}`, async ({ page }) => {
		// Relative to the console's address, which may carry a prefix (E2E_BASE_PATH).
		await page.goto(path.slice(1) || './');
		await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
		// The page keeps its event stream open, so it is never network-idle: wait
		// for the live job count instead, then for what the page loads.
		await expect(page.locator('header').getByText(/active jobs?/)).toBeVisible();
		await page.waitForTimeout(1500);
		const results = await new AxeBuilder({ page })
			.withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
			.analyze();
		const serious = results.violations
			.filter((v) => v.impact === 'serious' || v.impact === 'critical')
			.map((v) => `${v.id}: ${v.help} (${v.nodes.map((n) => n.target.join(' ')).join('; ')})`);
		expect(serious).toEqual([]);
	});
}
