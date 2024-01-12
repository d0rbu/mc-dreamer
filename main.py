import os
import torch as th
import time
from core.pmc import get_next_page_button, sign_in, project_urls
from core.browser import get_browser
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from dotenv import load_dotenv


PMC_URL = "https://www.planetminecraft.com"
PMC_SEARCH_URL = f"{PMC_URL}/projects/?mode=advanced&category%5B%5D=10&category%5B%5D=18&category%5B%5D=87&category%5B%5D=42&category%5B%5D=117&category%5B%5D=15&category%5B%5D=11&category%5B%5D=21&category%5B%5D=16&category%5B%5D=88&category%5B%5D=12&category%5B%5D=13&category%5B%5D=17&share%5B%5D=schematic&platform=1"
PMC_LOGIN_URL = "{PMC_URL}/account/sign_in/"

load_dotenv()

def scrape_pmc(
    output_dir: str | os.PathLike = "outputs",
    raw_output_dir: str | os.PathLike = "raw_outputs",
    driver_path: str | os.PathLike = os.path.join("drivers", "chromedriver.exe"),
) -> None:
    browser = get_browser(extensions=["ublock"], download_dir=raw_output_dir)

    sign_in(browser)

    browser.get(PMC_SEARCH_URL)

    while next_page_button := get_next_page_button(browser):  # lmao get fucked van rossum
        current_projects_page = browser.current_url
        for project_url in project_urls(browser):
            browser.get(project_url)
            download_button = browser.find_element(By.CLASS_NAME, "branded-download")
            download_button.click()

            wait = WebDriverWait(browser, 5)  # in case download opens a new tab or something
            wait.until(EC.title_contains("Minecraft Map"))

            browser.get(current_projects_page)
        
        next_page_button.click()

    browser.quit()


if __name__ == "__main__":
    scrape_pmc()
