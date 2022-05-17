# This file reads in the metadata CSV file and generates labels
# There are several different label sets that are generated, listed as follows:
#   Full:   Includes all data without consideration for viewpoint or date
#   Top:    Includes all top view data without consideration for date
#   Side:   Includes all side view data without consideration for date
#   "Year": Includes all data for "Year", where year is a int representing a timestamp year (eg 2022)

# IMPROVED: We can split the data into basic "Building Blocks". The blocks are as follows:
# Dimension 1: Time in Calender Year
# Dimension 2: Viewpoint, either Top, Left, or Right

import csv
import numpy as np

def geneerate_label(metadata_path):
    # metadata_path = "C:/Users/zpmao/Downloads/boem-belugas-runtime/data/metadata.csv"

    # Non Senariofied Label set
    x_full = []
    y_full = []

    # List of years present in the metadata
    years = []

    # List of the whales present
    whales = []

    with open(metadata_path) as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:

            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1

            else:
                x_full.append(row[0])
                y_full.append(row[7])

                cur_datapoint_year = row[5].split("-")
                if cur_datapoint_year[0] not in years:
                    years.append(cur_datapoint_year[0])

                cur_whale_id = row[7].split("e")
                y_full.append(cur_whale_id)
                if cur_whale_id[1] not in whales:
                    whales.append(cur_whale_id[1])


    print(x_full)
    print(y_full)
    print(years)
    print(whales)

    # Each entry in this 2D list is a list of image_ids that belong to each category (Thus making the overall structure 3D)
    # In the years dimension: the order of years is in the order they appear in the "years" list we created above.
    # ie years = [2017 2018] -> x_data_matrix[0] refers to the images from 2017
    # In the viewpoint dimension the following enumeration is observed
    #   0 -> Top
    #   1 -> Left
    #   2 -> Right
    # For a complete example:
    #   If years == [2017 2018]
    #   x_data_matrix[0][1] is all the image_ids from 2017 from a Left viewpoint
    #   x_data_matrix[2][0] is all the image_ids from 2018 from a a Top viewpoint

    # x_data_matrix[x][y
    # i is years dimension
    # j is viewpoint dimension
    x_data_matrix = t = [[[]] * 3 for i in range(len(years))]

    # Confirmation Printout
    print("There are: " + str(len(x_data_matrix)) + " Years")
    print("There are: " + str(len(x_data_matrix[0])) + " Viewing angles")


    # Second time we go through and append the image IDs to their respective blocks
    with open(metadata_path) as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:

            if line_count != 0:

                cur_datapoint_year = row[5].split("-")
                cur_viewpoint = row[4]

                cur_year_index = years.index(cur_datapoint_year[0])
                cur_viewpoint_num = -1

                if cur_viewpoint == "top":
                    cur_viewpoint_num = 0

                elif cur_viewpoint == "left":
                    cur_viewpoint_num = 1

                elif cur_viewpoint == "right":
                    cur_viewpoint_num = 2

                else:
                    print("Error: Viewpoint input was not \"top\", \"left\", or \"right\"")

                x_data_matrix[cur_year_index][cur_viewpoint_num].append(row[0])

            line_count += 1


    print(f'Processed {line_count} lines.')

    return years, x_data_matrix

