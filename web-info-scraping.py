
from selenium import webdriver
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
headers_filename = 'scrape/headers.txt'


csv_filename = os.path.join(output_directory, 'data.csv')


all_headers = {'patient id': False, 'hospital': False, 'age': False, 'gender': False, 'body temperature': False,
                   'underlying disease': False, 'is_covid': False,
                   'is_ct': False, 'morbidity': False, 'mortality': False}
other_info_header_set = False


def get_overview_info(bsoup):
    overview_table = bsoup.find('table', {"class": "array1"})
    table_rows = overview_table.findAll('tr')[1:]

    all_info = all_headers.copy()
    for table_row in table_rows:
        row_label = table_row.find('td', {'class': 'tablabel'}).text[:-2].lower()
        row_content = table_row.find('td', {'class': 'content'}).text

        if 'hospital' in row_label:
            all_info['hospital'] = row_content
        elif 'age' in row_label:
            all_info['age'] = row_content
        elif 'gender' in row_label:
            all_info['gender'] = row_content
        elif 'underlying' in row_label:
            all_info['underlying disease'] = row_content
        elif 'sars-cov-2' in row_label:
            all_info['is_covid'] = row_content
        elif 'computed' in row_label:
            all_info['is_ct'] = row_content
        elif 'morbidity' in row_label:
            all_info['morbidity'] = row_content
        elif 'mortality' in row_label:
            all_info['mortality'] = row_content

    return all_info


def get_other_info(bsoup, all_info):
    global other_info_header_set
    other_tables = bsoup.findAll('table')[1:]

    for other_table in other_tables:
        row_infos = other_table.findAll('tr')[1:]
        for row_info in row_infos:
            each_infos = row_info.findAll('td')
            name_abbreviation = each_infos[1].text
            value = each_infos[2].text

            all_info[name_abbreviation] = value

    return all_info


domain_url = 'http://ictcf.biocuckoo.cn/view.php?id='
def get_headers(patients):
    try:
        with open(headers_filename, 'r') as f:
            headers = f.read()

            for header in headers.split(","):
                if all_headers.get(header, None) is None:
                    all_headers[header] = False
            return
    except FileNotFoundError:
        print("header file not found, generating headers")

    for patient in patients:
        if 'Patient' not in patient:
            continue

        driver.get(domain_url + patient)
        bsoup = BeautifulSoup(driver.page_source, 'html.parser')

        other_tables = bsoup.findAll('table')[1:]

        for other_table in other_tables:
            row_infos = other_table.findAll('tr')[1:]
            for row_info in row_infos:
                each_infos = row_info.findAll('td')
                name_abbreviation = each_infos[1].text

                if not other_info_header_set:
                    if all_headers.get(name_abbreviation, None) is None:
                        all_headers[name_abbreviation] = False
        # remove this break
        # break


# just temporarily
all_patients = os.listdir('scrape/')
all_patients = sorted(all_patients)
get_headers(all_patients)

# convert dictionary to list to write csv file
# write the first row as header first
headers = list(all_headers.keys())
with open(csv_filename, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

start = False
for patient in all_patients:
    if 'Patient' not in patient:
        continue

    if patient == 'Patient 1152':
        start = True

    if start:
        driver.get(domain_url + patient)
        bsoup = BeautifulSoup(driver.page_source, 'html.parser')
        all_info = get_overview_info(bsoup)
        all_info = get_other_info(bsoup, all_info)
        all_info['patient id'] = patient

        with open(csv_filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(list(all_info.values()))

            # save image
            # patient_path = os.path.join(output_directory, patient)
            # os.makedirs(patient_path, exist_ok=True)



