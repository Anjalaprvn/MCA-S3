
n = int(input("Enter the number of students: "))
marks = []
for i in range(n):
    mark = float(input(f"Enter marks for student {i + 1}: "))
    marks.append(mark)
total = sum(marks)
average = total / n if n > 0 else 0
print(f"Total Marks: {total}")
print(f"Average Marks: {average:.2f}")

output:

Enter the number of students: 3
Enter marks for student 1: 32
Enter marks for student 2: 45
Enter marks for student 3: 35
Total Marks: 112.0
Average Marks: 37.33
