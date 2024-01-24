import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.remote.webelement import WebElement


def project_urls(
    browser: webdriver.Chrome,
) -> list[str]:
    projects = browser.find_elements(By.CLASS_NAME, "resource")
    urls = []

    for project in projects:
        if project.get_attribute("data-type") == "resource":
            project_info = project.find_element(By.CLASS_NAME, "r-info")
            title = project_info.find_element(By.CLASS_NAME, "r-title")
            urls.append(title.get_attribute("href"))
    
    return urls

def get_next_page_button(browser: webdriver.Chrome) -> WebElement | None:
    try:
        next_page_button = browser.find_element(By.CLASS_NAME, "pagination_next")

        if "holder" in next_page_button.get_attribute("class"):
            return None

        return next_page_button
    except:
        return None

def sign_in(browser: webdriver.Chrome) -> None:
    browser.get("https://www.planetminecraft.com/account/sign_in/")

    email = os.getenv("PMC_EMAIL")
    password = os.getenv("PMC_PASSWORD")

    email_input = browser.find_element(By.ID, "email")
    email_input.send_keys(email)
    password_input = browser.find_element(By.ID, "password")
    password_input.send_keys(password)
    password_input.send_keys(Keys.RETURN)

    try:
        wait = WebDriverWait(browser, 3)
        wait.until(EC.title_contains("Planet Minecraft"))  # wait until we are redirected to homepage
    except:
        email_input = browser.find_element(By.ID, "email")
        email_input.send_keys(email)
        password_input = browser.find_element(By.ID, "password")
        password_input.send_keys(password)
        print("HELP ME SIGN IN!!!")

        wait = WebDriverWait(browser, float("inf"))
        wait.until(EC.title_contains("Planet Minecraft"))