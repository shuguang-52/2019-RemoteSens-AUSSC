# -*- coding: utf-8 -*-
import os
import requests


def downloader(url, path):
    size = 0
    response = requests.get(url, stream=True)
    chunk_size = 1024
    content_size = int(response.headers['content-length'])
    if response.status_code == 200:
        print('[Size of HSI data set]: %0.3f MB' % (content_size / chunk_size / 1024))
        with open(path, 'wb') as file:
            for data in response.iter_content(chunk_size=chunk_size):
                file.write(data)
                size += len(data)
                print('\r' + '[Download progress]: %s%.2f%%'
                      % ('>' * int(size * 50 / content_size), float(size / content_size * 100)), end='')


hsi_url = ['http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
           'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',
           'http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
           'http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat',
           'http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
           'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat']

for i in range(len(hsi_url)):
    file_name = hsi_url[i].split('/')[-1]
    data_path = 'datasets/' + str(file_name)
    if os.path.exists(data_path) == False:
        print("Downloading data file from %s to %s" % (hsi_url[i], data_path))
        downloader(hsi_url[i], data_path)
        print('\n'+str(file_name) + " is Successfully downloaded")
    else:
        print(str(file_name) + " already exists")

print('All HSI dataset have already existed!')
