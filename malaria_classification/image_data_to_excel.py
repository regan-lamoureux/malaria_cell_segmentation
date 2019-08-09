import os
import openpyxl

def excel(workbook, imageName, feat_list):
    global counter
    xfile = openpyxl.load_workbook(workbook)
    xfile.sheetnames  # all names
    sheet = xfile['Sheet1']


    for i in range(0, len(feat_list)):
        line = alphabet[i] + str(counter)
        sheet[line] = feat_list[i]

    xfile.save(workbook)

def image_to_excel(my_path, workbook):
    global counter
    folder = os.fsdecode(my_path)
    # return os.listdir(folder)
    # for image in os.listdir(folder):
    for i in range(0, 344):
        # print(str(counter) + '. ' + image)
        if os.listdir(folder)[i] == '.DS_Store':
            pass
        else:
            ret = feature_extraction(my_path + '/' + os.listdir(folder)[i])
            excel(workbook, os.listdir(folder)[i], ret)
            counter += 1
