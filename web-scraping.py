
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import wget
import csv
import os
from urllib.error import HTTPError

start_index = 41
output_directory = 'scrape'
URL = 'http://ictcf.biocuckoo.cn/Resource.php'
domain = 'http://ictcf.biocuckoo.cn/'
driver = webdriver.Chrome('/Users/darylfung/chromedriver')


csv_filename = os.path.join(output_directory, 'data.csv')
# csv_header = ['patient', 'age', 'gender', 'covid', 'ct', 'morbidity', 'mortality']
# with open(csv_filename, 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(csv_header)

for current_page in range(start_index, 78):
    driver.get(URL)
    driver.execute_script(f'Submit({current_page})')
    os.makedirs(output_directory, exist_ok=True)

    bsoup = BeautifulSoup(driver.page_source, 'html.parser')
    all_table_rows = bsoup.findAll('tr')

    for i in range(9, len(all_table_rows)-1):
        current_row = all_table_rows[i]
        image_page = domain + current_row.find('a').attrs['href']
        # get image link
        driver.get(image_page)
        element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "CT")))

        image_bsoup = BeautifulSoup(driver.page_source, 'html.parser')
        images_link = image_bsoup.find(id='CT').findAll('img')
        # save information
        info = bsoup.findAll('tr')[i].findAll('td')
        patient = info[1].text
        age = info[2].text
        gender = info[3].text
        is_covid = info[4].text
        is_ct = info[5].text
        severity_score = info[6].text
        is_cured = info[7].text
        row_data = [patient, age, gender, is_covid, is_ct, severity_score, is_cured]

        if is_ct == 'N/A':
            continue

        with open(csv_filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

        # save image
        patient_path = os.path.join(output_directory, patient)
        os.makedirs(patient_path, exist_ok=True)

        try:
            for index, image_link in enumerate(images_link):
                current_image = domain + image_link.attrs['src']
                wget.download(current_image, out=os.path.join(patient_path, f'{index}.jpg'))
        except HTTPError:
            print('image not found')
