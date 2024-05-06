from playwright.sync_api import sync_playwright
import asyncio

async def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        await page.goto("https://www.youtube.com/watch?v=ZM2Bghka_p8")

        # Scroll down to load comments
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

        # Wait for comments to load
        await page.wait_for_selector("ytd-comment-renderer #content-text")

        comments = page.query_selector_all("ytd-comment-renderer #content-text")
        for comment in comments:
            print(comment.text_content())

        await browser.close()

# Run the asyncio event loop
asyncio.run(main())