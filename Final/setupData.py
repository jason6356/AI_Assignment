import csv

labels = []
courses = []
courseName = []
courseDictionary = {}

def init():
    with open("datasets/course.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count  == 0:
                labels = row
            else:
                #print(row)
                courseName.append(row[0])
                course = {
                    labels[0] : row[0],
                    labels[1] : row[1],
                    labels[2] : row[2],
                    labels[3] : row[3],
                    labels[4] : row[4],
                    labels[5] : row[5],
                    labels[6] : row[6]
                }
                courses.append(course)
                courseDictionary[row[0]] = course
                
                
            line_count+=1
    
    return labels, courses, courseName, courseDictionary