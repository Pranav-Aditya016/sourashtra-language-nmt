"""
Capture Website Screenshots for IEEE Paper
============================================
Takes screenshots of the Sourashtra Translator web application
for inclusion in the IEEE paper.

Requirements:
    pip install selenium Pillow
    
Make sure:
    1. The Flask server is running (python app.py)
    2. Chrome browser is installed

Usage:
    python capture_screenshots.py
"""

import os
import time

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    print("Installing selenium...")
    os.system("pip install selenium")
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "ieee_paper")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = "http://localhost:5000"


def setup_driver():
    """Create a Chrome driver with appropriate settings."""
    opts = Options()
    opts.add_argument("--window-size=1280,900")
    opts.add_argument("--force-device-scale-factor=1.5")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    # Remove headless to see the browser; add it for CI
    # opts.add_argument("--headless=new")
    
    driver = webdriver.Chrome(options=opts)
    driver.implicitly_wait(5)
    return driver


def screenshot(driver, filename, full_page=False):
    """Take a screenshot and save to ieee_paper/ folder."""
    path = os.path.join(OUTPUT_DIR, filename)
    time.sleep(1)  # Wait for animations
    
    if full_page:
        # Get full page height
        total_height = driver.execute_script("return document.body.scrollHeight")
        driver.set_window_size(1280, total_height + 200)
        time.sleep(0.5)
    
    driver.save_screenshot(path)
    print(f"  ✓ Saved: {path}")
    
    if full_page:
        driver.set_window_size(1280, 900)


def capture_translate_tab(driver):
    """Screenshot 1: Translate tab with a result."""
    print("\n[1] Translate tab — typing 'water' ...")
    driver.get(BASE_URL)
    time.sleep(2)
    
    # Type a word and translate
    input_box = driver.find_element(By.ID, "sourceInput")
    input_box.clear()
    input_box.send_keys("water")
    
    # Click translate
    btn = driver.find_element(By.ID, "translateBtn")
    btn.click()
    time.sleep(2)
    
    screenshot(driver, "webapp_translate.png")


def capture_translate_tamil(driver):
    """Screenshot 2: Translate tab with Tamil input."""
    print("\n[2] Translate tab — Tamil mode ...")
    driver.get(BASE_URL)
    time.sleep(2)
    
    # Switch to Tamil
    tamil_btn = driver.find_element(By.CSS_SELECTOR, '[data-lang="tamil"]')
    tamil_btn.click()
    time.sleep(0.5)
    
    input_box = driver.find_element(By.ID, "sourceInput")
    input_box.clear()

    # Use JS to set Tamil text (keyboard may not support it)
    driver.execute_script(
        'document.getElementById("sourceInput").value = "அம்மா"'
    )
    time.sleep(0.3)
    
    btn = driver.find_element(By.ID, "translateBtn")
    btn.click()
    time.sleep(2)
    
    screenshot(driver, "webapp_translate_tamil.png")


def capture_dictionary_tab(driver):
    """Screenshot 3: Dictionary browser."""
    print("\n[3] Dictionary tab ...")
    driver.get(BASE_URL)
    time.sleep(2)
    
    # Click Dictionary tab
    dict_tab = driver.find_element(By.CSS_SELECTOR, '[data-tab="dictionary"]')
    dict_tab.click()
    time.sleep(2)
    
    screenshot(driver, "webapp_dictionary.png")


def capture_about_tab(driver):
    """Screenshot 4: About page."""
    print("\n[4] About tab ...")
    driver.get(BASE_URL)
    time.sleep(2)
    
    about_tab = driver.find_element(By.CSS_SELECTOR, '[data-tab="about"]')
    about_tab.click()
    time.sleep(2)
    
    screenshot(driver, "webapp_about.png")


def main():
    print("=" * 55)
    print("  Capturing Website Screenshots for IEEE Paper")
    print("=" * 55)
    
    driver = setup_driver()
    
    try:
        capture_translate_tab(driver)
        capture_translate_tamil(driver)
        capture_dictionary_tab(driver)
        capture_about_tab(driver)
    finally:
        driver.quit()
    
    print("\n" + "=" * 55)
    print("  All screenshots saved to ieee_paper/")
    print("=" * 55)


if __name__ == "__main__":
    main()
