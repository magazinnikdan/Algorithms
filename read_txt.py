lines = []
with open('OFFICE.txt') as f:
    lines = f.readlines()

count = 0
for line in lines:
    count += 1
    line_data = line.split(',')
    print(line_data[0],line_data[1])