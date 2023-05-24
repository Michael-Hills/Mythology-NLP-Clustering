import requests
import json
import jsonlines
import time
import os
from bs4 import BeautifulSoup


def checkFiles():
    """
    Function to check the contaol file exists
    """
    start = 1
    if(os.path.isfile(os.getcwd() + "\control.json")):
        print("path exists")
        with open('control.json') as fp:
            parameters = json.load(fp)
            current = parameters['current']
            if current == "A'akuluujjusi":
                start = 0

    return start,current

def scrapeInfoBox(soup):
    """
    Function to get the information within the summary tables
    """
    info = []
    box = soup.find('aside', class_="portable-infobox pi-background pi-border-color pi-theme-wikia pi-layout-default")
    role = box.find_all('h2', {"class":"pi-item pi-item-spacing pi-title pi-secondary-background"})

    #gets the role of the myths
    for i in range(len(role)):
        role[i] = role[i].getText()
    
    #if role exists, add role to info list
    if role != None:
        role = " ".join(role)
        info.append({'role': role})

    #gets all rows within the table
    information = box.find_all('div', class_="pi-item pi-data pi-item-spacing pi-border-color")
    
    for data in information:
        #checks the rows header
        section = data.parent
        head = section.find('h2', class_ ="pi-item pi-header pi-secondary-font pi-item-spacing pi-secondary-background")
        if (head != None):
            #discards information if it is not deemed important
            header = head.get_text()
            if header not in ["Languages","Relationships"]:
                #add labels and values to info list
                label = data.find('h3', class_="pi-data-label pi-secondary-font").get_text()
                value = data.find("div", class_= "pi-data-value pi-font").get_text(separator=" ")
                info.append({label : value})
        else:
            label = data.find('h3', class_="pi-data-label pi-secondary-font").get_text()
            value = data.find("div", class_= "pi-data-value pi-font").get_text(separator=" ")
            info.append({label : value})
            
    
    return info

def scrapeCategories(soup):
    """
    Function to get the categories of each page
    """
    categories = []
    #scrapes all the headers
    header = soup.find('div',class_="page-header__categories")
    if (header != None):
        links = header.find_all('a')
        if (len(links) != 0):
            for link in links:
                if (link.has_attr('class')):
                    if (link['class'][0] != "wds-dropdown__toggle"):
                        categories.append(link.get_text())
                else:
                    categories.append(link.get_text())

    return categories

def scrapeMainText(url):
    """
    Function to get the main text of each myth
    """
    mainText = []
    info = {}
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    content = soup.find_all(id = 'mw-content-text')[0]
    title = soup.select('head title')[0].get_text().split('|')[0] #gets the title
    title = title[:len(title)-1]
    text = content.find_all(['p','li'])

    #text categories to not collect
    notCollect = ['relations','family','see also','external','gallery','sources','citations',
                  'references','film','video Games','tv shows']
   
    info['title'] = title
    
    for item in text:
        add = True
        headers = []
        headers.append(item.parent.find_previous_sibling(['h2','h3']))
        headers.append(item.find_previous_sibling(['h2','h3']))

        #ignore any tags with a class (i.e. ads, tables)
        if (item.has_attr('class')):
            add = False
        
        #stops collecting the table within the main text
        elif (item.find(class_="portable-infobox pi-background pi-border-color pi-theme-wikia pi-layout-default") != None):
            add = False
            

        else:
            for parent in item.parents:
                if (parent.has_attr('class')):
                    if(parent['class'][0] in ["portable-infobox", "references"]):
                        add = False
                       
        #ignore any text with unimportant headers
        if (len(headers) != 0):
            for header in headers:
                if header != None:
                    if (any(notCol for notCol in notCollect if (notCol in header.get_text().lower()))):
                        add = False
                        

        
        if add == True:
            if (item.get_text() != "\n"):
                mainText.append(item.get_text().strip('\n'))

    
    info['text'] = " ".join(mainText)
    table = content.find('aside', {"class": "portable-infobox pi-background pi-border-color pi-theme-wikia pi-layout-default"})
    if table:
        tableInfo = scrapeInfoBox(soup)
        info['table'] = tableInfo

    categories = scrapeCategories(soup)
    info['categories'] = categories

    return info



def writeToJson():

    notCollect = ['Gallery','disambiguation']

    while True:

        start,current = checkFiles()

        #creates GET request to web server
        page = requests.get("https://mythus.fandom.com/wiki/Special:AllPages?from=" + current)

        #create instance of BeautifulSoup class
        soup = BeautifulSoup(page.content, 'html.parser')

        #create list containing all links to pages 
        mythsPage = soup.find_all('ul', class_='mw-allpages-chunk')
        myths = mythsPage[0].find_all('a')
        mythInfo = []

        #collect myths 15 at a time
        for myth in myths[start:15]:
            current = myth['title']
            if (not myth.has_attr('class')):
                if (all(x not in current for x in notCollect)):
                    pageUrl = myth['href']
                    mythInfo.append(scrapeMainText("https://mythus.fandom.com/" + pageUrl))
                    time.sleep(0.5)
    
        #add to the json lines file
        with jsonlines.open('data.jsonl', mode='a') as writer:
            for i in range(len(mythInfo)):
                writer.write(mythInfo[i])   
                print(mythInfo[i]['title'], " - successfully added")
            
        #update the control file with most recent myth
        with open('control.json', 'w') as f:
            json.dump({"current":current}, f)


        if current == "\ud83e\uddde\u200d\u2642\ufe0f":
            break

        

        
writeToJson()




