import requests
import csv
from bs4 import BeautifulSoup


url_list = []

# Open the text file containing URLs
with open('TARUMT.txt', 'r') as file:
    # Read each line of the file and append it to the list
    for line in file:
        url = line.strip()
        if url and url != 'None':
            url_list.append(url)
        #print(line.strip())

# Print the list of URLs
#print(url_list)

courses = []

for url in url_list:
    if url is None:
        continue

    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    result = []
    courseName = soup.select_one('#section-start-journey > div > h4')
    programmeOverview = soup.select_one('#tabs-1 > p:nth-child(2)')
    duration = soup.select_one('#tabs-1 > p:nth-child(4)')
    img = soup.select_one('#tabs-1 > p:nth-child(6) > img')
    tab7 = soup.select_one('#tabs-7')
    branch = soup.select_one('#tabs-1 > p:nth-child(5)')
    tab6 = soup.select_one('#tabs-6')

    nav_tabs = soup.select_one('#tabs > ul')

    if nav_tabs:
        print(nav_tabs)

    if courseName:
        #print(courseName.getText())
        result.append(courseName.getText())
    else:
        result.append(None)
    if programmeOverview:
        #print(programmeOverview.getText()) 
        result.append(programmeOverview.getText())
    else:
        result.append(None)
    if duration:
        result.append(duration.getText())
        #print(duration.getText())
    else:
        result.append(None)
    if img:
        #print(img['src'])
        result.append(img['src'])
    else:
        result.append(None)

    if branch:
        result.append(branch.getText())
    else:
        result.append(None)

    if tab6:
        fee = tab6.find_all('p')    
        for f in fee:
            if f:
                #print(f.getText())
                result.append(f.getText())

    if tab7:
        fee = tab7.select_one('p:nth-child(2)')    
        if fee:
            #print(fee.getText())
            result.append(fee.getText())
        else:
            result.append(None)
    else:
        result.append(None)

    courses.append(result)

with open('course.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for course in courses: 
        writer.writerow(course)
#            writer.writerow([courseName.getText(),programmeOverview.getText(),duration.getText()])

# page = requests.get(url)


# ##print(page.text)

# soup = BeautifulSoup(page.content, "html.parser")

# results = soup.find_all('a')

# print(results)
# print(results.beautify())
