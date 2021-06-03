import os
import pandas as pd
import csv
import random
import itertools


BASE_DIR = "/Users/augustolimonti/Desktop/753_Project_Final/Dog_Dataset"
folders = os.listdir(BASE_DIR)
folders.remove(".DS_Store")


with open("smaller_train.csv", "w+", newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['names', 'images', 'labels'])
    for files in folders:
        path = os.path.join(BASE_DIR, files)
        label = folders.index(files)
        for root, dirs, files in os.walk(path):
            for filename in files:
                thewriter.writerow([folders[label], filename, label])
    f.close()

main_list = pd.read_csv('smaller_train.csv')
training_names = list(main_list['names'])
training_imgs = list(main_list['images'])
training_labels = list(main_list['labels'])
main_list = pd.DataFrame({'Name' : training_names, 'Images': training_imgs,'Breed': training_labels})
main_list.Images = main_list.Images.astype(str)
main_list.Breed = main_list.Breed.astype(str)
main_list.to_csv('smaller_train.csv')
training_set = pd.read_csv("smaller_train.csv")
# training_set = main_list.sample(frac=1)
# training_set = main_list.drop(main_list.index[5150:20586])
training_set = main_list.drop(main_list.index[5150:20586])
training_set = training_set.sample(frac=1)
training_set.to_csv('smaller_train.csv')

with open("smaller_train.csv", "rt") as f, open("smaller_test.csv", "w+", newline= '') as t:
    thewriter = csv.writer(t)
    csv_input = csv.reader(f)
    thewriter.writerows(itertools.islice(csv_input, 0, 650))
    new_training_set = training_set.drop(training_set.index[0:650])
    # thewriter.writerows(itertools.islice(csv_input, 0, 150))
    # new_training_set = training_set.drop(training_set.index[0:150])
    # # thewriter.writerows(itertools.islice(csv_input, 0, 150))
    # # new_training_set = training_set.drop(training_set.index[0:150])
    t.close()
    f.close()

with open("smaller_test.csv", 'r', newline='') as f, open("test_results.csv", 'w', newline='') as t:
    reader = csv.reader(f)
    writer = csv.writer(t)
    for row in reader:
        writer.writerow(row)
    # t.close()
    # f.close()

# test_results = pd.read_csv("test_results.csv")
# test_results = test_results.drop(test_results.index[0:1])
# test_results.to_csv('test_results.csv')

testing_set = pd.read_csv("smaller_test.csv")
testing_imgs = list(testing_set['Images'])
testing_labels = list(testing_set['Breed'])
testing_set = pd.DataFrame({'Images': testing_imgs,'Breed': testing_labels})
testing_set.Images = testing_set.Images.astype(str)
testing_set.Breed = testing_set.Breed.astype(str)
testing_set.to_csv('smaller_test.csv')
testing_set.drop(["Breed"], axis = 1, inplace = True)

new_training_set.to_csv('smaller_train.csv')
testing_set.to_csv('smaller_test.csv')
# print(new_training_set.head())
# print(testing_set.head())
#
# with open("test_results.csv", 'r') as f:
#     csv_input = csv.DictReader(f)
#     for row in csv_input:
#         print(row['Breed'])
