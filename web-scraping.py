
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import wget
import csv
import os
from urllib.error import HTTPError

start_index = 1
output_directory = 'scrape'
URL = 'http://ictcf.biocuckoo.cn/Resource.php'
domain = 'http://ictcf.biocuckoo.cn/'
driver = webdriver.Chrome('/Users/darylfung/chromedriver')


csv_filename = os.path.join(output_directory, 'data.csv')


overview_header = ['patient id', 'hospital', 'age', 'gender', 'body temperature', 'underlying disease', 'is_covid', 'is_ct', 'morbidity', 'mortality']
other_info_header_set = False
other_info_header = []


def get_overview_info(bsoup):
    overview_table = bsoup.find('table', {"class": "array1"})
    table_rows = overview_table.findAll('tr')[1:]

    row_info = []
    for table_row in table_rows:
        row_label = table_row.find('td', {"class": "tablabel"})
        row_content = table_row.find('td', {"class": "content"})

        row_info.append(row_content.text)
    return row_info


def get_other_info(bsoup):
    global other_info_header_set
    other_tables = bsoup.findAll('table')[1:]

    all_other_info = []
    for other_table in other_tables:
        row_infos = other_table.findAll('tr')[1:]
        for row_info in row_infos:
            each_infos = row_info.findAll('td')
            name_abbreviation = each_infos[1].text
            value = each_infos[2].text
            all_other_info.append(value)

            if not other_info_header_set:
                other_info_header.append(name_abbreviation)

    if not other_info_header_set:
        other_info_header_set = True
        with open(csv_filename, 'a') as f:
            writer = csv.writer(f)
            headers = overview_header + other_info_header
            writer.writerow(headers)

    return all_other_info


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

        # find all table informations in the patient
        overview_info = get_overview_info(image_bsoup)

        # get all other information
        other_info = get_other_info(image_bsoup)

        images_link = image_bsoup.find(id='CT').findAll('img')
        # save information
        info = bsoup.findAll('tr')[i].findAll('td')
        patient = info[1].text
        is_ct = info[5].text
        overview_info = [patient] + overview_info
        all_info = overview_info + other_info

        if is_ct == 'N/A':
            continue

        with open(csv_filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(all_info)

        # save image
        patient_path = os.path.join(output_directory, patient)
        os.makedirs(patient_path, exist_ok=True)

        try:
            for index, image_link in enumerate(images_link):
                current_image = domain + image_link.attrs['src']
                wget.download(current_image, out=os.path.join(patient_path, f'{index}.jpg'))
        except HTTPError:
            print('image not found')


