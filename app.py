# =============================
# ✅ Install and Import Libraries
# =============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import streamlit as st
import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import urllib.request

# =============================
# ✅ Setup Selenium Driver
# =============================
def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(15)
    return driver

# =============================
# ✅ Web Scraping Functions
# =============================
def get_page_load_speed(driver, url):
    start_time = time.time()
    try:
        driver.get(url)
    except:
        return 99
    return round(time.time() - start_time, 2)

def get_image_metrics(driver):
    images = driver.find_elements(By.TAG_NAME, "img")
    total_images = len(images)
    lazy_loaded = sum(1 for img in images if img.get_attribute("loading") == "lazy")
    return total_images, lazy_loaded

def check_navigation(driver):
    try:
        cta_buttons = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, "//button[contains(text(), 'Get Started') or contains(text(), 'Buy') or contains(text(), 'Subscribe')]"))
        )
    except:
        cta_buttons = []

    menu = driver.find_elements(By.TAG_NAME, "nav")
    breadcrumbs = driver.find_elements(By.CLASS_NAME, "breadcrumb")

    return len(cta_buttons) > 0, len(menu) > 0, len(breadcrumbs) > 0

def get_link_metrics(driver):
    links = driver.find_elements(By.TAG_NAME, "a")
    total_links = len(links)
    broken_links = sum(1 for link in links if not link.get_attribute("href"))
    return total_links, broken_links

def check_mobile_responsiveness(driver, url):
    driver.set_window_size(375, 812)
    driver.get(url)
    time.sleep(2)
    return driver.execute_script("return document.body.scrollWidth <= 375")

# =============================
# ✅ Load Model from GitHub
# =============================
def load_model_from_github(url):
    model_path = 'random_forest_model.pkl'
    try:
        urllib.request.urlretrieve(url, model_path)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model from GitHub: {e}")
        return None

# Use the updated GitHub URL
github_model_url = 'https://raw.githubusercontent.com/saurabhkhare2024/uxfeature/main/random_forest_model.pkl'
model = load_model_from_github(github_model_url)

# =============================
# ✅ Streamlit App
# =============================
st.title('UX Score Predictor')
st.write('Enter a URL to evaluate its user experience score.')

url_input = st.text_input('Enter URL:')

if st.button('Predict UX Score'):
    driver = setup_driver()
    
    try:
        load_speed = get_page_load_speed(driver, url_input)
        total_images, lazy_loaded = get_image_metrics(driver)
        cta_present, menu_present, breadcrumbs_present = check_navigation(driver)
        total_links, broken_links = get_link_metrics(driver)
        mobile_friendly = check_mobile_responsiveness(driver, url_input)

        if model is None:
            st.error('❌ Model could not be loaded. Please check the GitHub URL.')
        else:
            # Prepare Input Data
            input_data = pd.DataFrame({
                'Page Load Speed (s)': [load_speed],
                'Total Images': [total_images],
                'Lazy Loaded Images': [lazy_loaded],
                'CTA Present': [cta_present],
                'Navigation Menu Present': [menu_present],
                'Breadcrumbs Present': [breadcrumbs_present],
                'Total Links': [total_links],
                'Broken Links': [broken_links],
                'Mobile Friendly': [mobile_friendly]
            })

            # Predict UX Score
            prediction = model.predict(input_data)[0]
            st.success(f'Predicted UX Score: {prediction:.2f}')

    except Exception as e:
        st.error(f'❌ Error: {e}')
    finally:
        driver.quit()
