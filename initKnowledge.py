import csv

thisdict = {}
questions = []

with open("./datasets/lumosquestiongenerator.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count  == 0:
            print(f'Column names are {",".join(row)}')
        else:
            print(row)
            questions.append(row[1])
            thisdict[row[1]] = row[2]
        line_count+=1

print(thisdict)
print(questions)
    