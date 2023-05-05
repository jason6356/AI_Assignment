import re
import pandas as pd

df = pd.read_csv('course.csv')
url_list = []

# Open the text file containing URLs
with open('TARUMT.txt', 'r') as file:
    # Read each line of the file and append it to the list
    for line in file:
        url = line.strip()
        if url and url != 'None':
            url_list.append(url)
        #print(line.strip())

result=[]
for url in url_list:
            
    pattern = r"/(\w+)/programmes|programme/"

    match = re.search(pattern, url)
    if match:
        fafb = match.group(1)
        # print(fafb)  # Output: fafb
        result.append(fafb)

result.pop(2)
result.pop(4)


# Create a new column
new_column_data = result
df['Faculty'] = new_column_data

# Save the updated DataFrame back to the CSV file
df.to_csv('course.csv', index=False)

