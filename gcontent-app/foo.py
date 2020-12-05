from collections import defaultdict
import csv


user_lessons = defaultdict(list)
with open("./data/user-lessons.csv", mode='r') as infile:
    reader = csv.reader(infile)
    header = next(reader)
    for row in reader:
        user_lessons[row[0]].append(row[1])

print(user_lessons)
