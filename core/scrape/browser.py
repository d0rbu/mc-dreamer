import os
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service


def get_browser(
    extensions: list[str] = [],
    download_dir: str | os.PathLike | None = None,
) -> webdriver.Chrome:
    service = Service(executable_path=os.path.join("drivers", "chromedriver.exe"))
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    if download_dir is not None:
        prefs = {
            "download.default_directory": os.path.join(os.path.abspath(download_dir), "a")[:-1]  # ensure end slash
        }
        options.add_experimental_option("prefs", prefs)

    for extension in extensions:
        extension_path = os.path.join("extensions", f"{extension}.crx")
        options.add_extension(extension_path)

    browser = webdriver.Chrome(service=service, options=options)

    return browser