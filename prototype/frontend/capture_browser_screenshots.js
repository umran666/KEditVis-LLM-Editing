import puppeteer from "puppeteer";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function main() {
  console.log("Launching local Chrome browser...");
  const browser = await puppeteer.launch({
    headless: "new",
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu"],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1720, height: 1080 });

  console.log("Navigating to http://127.0.0.1:5173 ...");
  await page.goto("http://127.0.0.1:5173", { waitUntil: "networkidle0" });

  // 1. Initial State
  const shot1 = path.join(__dirname, "screenshot_1_initial.png");
  await page.screenshot({ path: shot1, fullPage: false });
  console.log(`Saved screenshot: ${shot1}`);

  // 2. Click "Compare" & "Edit"
  const compareBtn = await page.waitForSelector("button.btn-action:nth-child(2)");
  await compareBtn.click();
  await new Promise((r) => setTimeout(r, 1000));

  const editBtn = await page.waitForSelector("button.btn-primary");
  await editBtn.click();
  await new Promise((r) => setTimeout(r, 1200));

  // 3. Full Page Screenshot showing entire dashboard from top to bottom
  const shotFull = path.join(__dirname, "screenshot_full_dashboard.png");
  await page.screenshot({ path: shotFull, fullPage: true });
  console.log(`Saved screenshot: ${shotFull}`);

  await browser.close();
  console.log("ALL SCREENSHOTS CAPTURED SUCCESSFULLY!");
}

main().catch((err) => {
  console.error("Error capturing browser screenshots:", err);
  process.exit(1);
});
