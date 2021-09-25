import heapq as hq

# list of students
list_stu = [(4, 10, 4, 10, 'Lucy'), (5, 10, 4, 100, 'Rina')]

# Arrange based on the roll number
hq.heapify(list_stu)

print("The order of presentation is :")

for i in list_stu:
    print(i)
