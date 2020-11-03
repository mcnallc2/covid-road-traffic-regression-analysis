import urllib.request
from bs4 import BeautifulSoup

f = open("./site_location_ids.txt", "r")


def get_location_ids():
    line = "-"
    location_ids = []
    while line != "":
        line = f.readline()
        if line.split("-")[0] == "NRA ":
            location_id = (line.split("-")[1].strip())
            location_ids.append(location_id)
    return location_ids


LOCATIONS = get_location_ids()
#  jan - aug ... data incomplete in october
MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]

nationwide_traffic = {
    "01": [],
    "02": [],
    "03": [],
    "04": [],
    "05": [],
    "06": [],
    "07": [],
    "08": [],
    "09": [],
}


for i, loc in enumerate(LOCATIONS):
    progress = (i/len(LOCATIONS)*100)
    print("-------------------------\n=> Location", loc, " - ", progress, "%")
    for j, month in enumerate(MONTHS):
        print("===> Month: ", month)
        # build the url with loc and month
        url = "https://www.nratrafficdata.ie/c2/tfmonthreport.asp?sgid=ZvyVmXU8jBt9PJE$c7UXt6&spid=NRA_" + \
            loc + "&reportdate=2020-" + month + "-01&enddate=2020-"+month+"-01"
        # scrape
        page = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(page, "html.parser")

        row024 = soup.find_all("tr")[2:][28]
        month_row = row024.find_all("td")

        count = 0
        daily_traffic = []
        for day in month_row:
            #  ignore 0-24, and total fields
            if count != len(month_row) - 1 and count != 0:
                if day.string.strip() == "-":  #  convert empty entries to 0
                    daily_traffic.append(0)
                else:
                    daily_traffic.append(int(day.string))
            count = count+1

        #  if empty initialise, if not add
        if nationwide_traffic[month] == []:
            nationwide_traffic[month] = daily_traffic
        else:
            nationwide_traffic[month] = nationwide_traffic[month] + \
                daily_traffic
    print("--> Reading next location.\n")


# this nationwide_traffic dictionary is the total traffic for each day
#  ...stored based on month
print("NATIONWIDE: ", nationwide_traffic)
