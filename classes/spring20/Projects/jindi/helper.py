import os
import xlsxwriter
import csv

def csv2xlsx():
    path = './results.csv'
    workbook = xlsxwriter.Workbook('result.xlsx')
    worksheet = workbook.add_worksheet()

    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        i = 0
        for row in reader:
            print(row)
            for col in range(len(row)):
                worksheet.write(i, col, row[col])
            i += 1

    workbook.close()


if __name__ == "__main__":
    csv2xlsx()
    # path = 'data/celeba/images'

    # print(len(os.listdir(path)))
    # for file in os.listdir(path):
    #     index = int(file.split('.')[0])
    #     if index > 50000:
    #         filepath = os.path.join(path, file)
    #         os.remove(filepath)
    
    