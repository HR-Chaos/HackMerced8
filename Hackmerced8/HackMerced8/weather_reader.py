import matplotlib.pyplot as plt
import csv

# SX02 - Maximum soil temperature with unknown cover at 10 cm depth
# WT03 - Thunder
# WT04 - Ice pellets, sleet, snow pellets, or small hail"
# PRCP - Precipitation
# WT05 - Hail (may include small hail)
# WSFM - Fastest mile wind speed
# WT06 - Glaze or rime
# WT07 - Dust, volcanic ash, blowing dust, blowing sand, or blowing obstruction
# WT08 - Smoke or haze
# SNWD - Snow depth
# WT09 - Blowing or drifting snow
# SX03 - Maximum soil temperature with unknown cover at 20 cm depth
# WDF1 - Direction of fastest 1-minute wind
# WDF2 - Direction of fastest 2-minute wind
# WDF5 - Direction of fastest 5-second wind
# WT10 - Tornado, waterspout, or funnel cloud"
# PGTM - Peak gust time
# WT11 - High or damaging winds
# TMAX - Maximum temperature
# WT13 - Mist
# FRGT - Top of frozen ground layer
# WSF2 - Fastest 2-minute wind speed
# FMTM - Time of fastest mile or fastest 1-minute wind
# ACMH - Average cloudiness midnight to midnight from manual observations
# WSF5 - Fastest 5-second wind speed
# SNOW - Snowfall
# WDFG - Direction of peak wind gust
# WT14 - Drizzle
# ACSH - Average cloudiness sunrise to sunset from manual observations
# WT16 - Rain (may include freezing rain, drizzle, and freezing drizzle)"
# WT18 - Snow, snow pellets, snow grains, or ice crystals
# WSF1 - Fastest 1-minute wind speed
# AWND - Average wind speed
# WT21 - Ground fog
# WSFG - Peak gust wind speed
# WT22 - Ice fog or freezing fog
# WT01 - Fog, ice fog, or freezing fog (may include heavy fog)
# WESD - Water equivalent of snow on the ground
# WT02 - Heavy fog or heaving freezing fog (not always distinguished from fog)
# SN03 - Minimum soil temperature with unknown cover at 20 cm depth
# PSUN - Daily percent of possible sunshine for the period
# TAVG - Average Temperature.
# TMIN - Minimum temperature
# SN02 - Minimum soil temperature with unknown cover at 10 cm depth
# WDFM - Fastest mile wind direction
# TSUN - Total sunshine for the period


titles = ["STATION","NAME","LATITUDE","LONGITUDE","ELEVATION","DATE","ACMH","ACMH_ATTRIBUTES","ACSH","ACSH_ATTRIBUTES","AWND","AWND_ATTRIBUTES","FMTM","FMTM_ATTRIBUTES","FRGT","FRGT_ATTRIBUTES","PGTM","PGTM_ATTRIBUTES","PRCP","PRCP_ATTRIBUTES","PSUN","PSUN_ATTRIBUTES","SN02","SN02_ATTRIBUTES","SN03","SN03_ATTRIBUTES","SNOW","SNOW_ATTRIBUTES","SNWD","SNWD_ATTRIBUTES","SX02","SX02_ATTRIBUTES","SX03","SX03_ATTRIBUTES","TAVG","TAVG_ATTRIBUTES","TMAX","TMAX_ATTRIBUTES","TMIN","TMIN_ATTRIBUTES","TSUN","TSUN_ATTRIBUTES","WDF1","WDF1_ATTRIBUTES","WDF2","WDF2_ATTRIBUTES","WDF5","WDF5_ATTRIBUTES","WDFG","WDFG_ATTRIBUTES","WDFM","WDFM_ATTRIBUTES","WESD","WESD_ATTRIBUTES","WSF1","WSF1_ATTRIBUTES","WSF2","WSF2_ATTRIBUTES","WSF5","WSF5_ATTRIBUTES","WSFG","WSFG_ATTRIBUTES","WSFM","WSFM_ATTRIBUTES","WT01","WT01_ATTRIBUTES","WT02","WT02_ATTRIBUTES","WT03","WT03_ATTRIBUTES","WT04","WT04_ATTRIBUTES","WT05","WT05_ATTRIBUTES","WT06","WT06_ATTRIBUTES","WT07","WT07_ATTRIBUTES","WT08","WT08_ATTRIBUTES","WT09","WT09_ATTRIBUTES","WT10","WT10_ATTRIBUTES","WT11","WT11_ATTRIBUTES","WT13","WT13_ATTRIBUTES","WT14","WT14_ATTRIBUTES","WT16","WT16_ATTRIBUTES","WT18","WT18_ATTRIBUTES","WT21","WT21_ATTRIBUTES","WT22","WT22_ATTRIBUTES"]

index = titles.index("PRCP")
data = [[] for i in range(len(titles))]
with open('Weather_data_Fresno.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        for i in range(len(titles)):
            data[i].append(row[titles[i]])
            
# print shape of data
# print("Shape of data: ", len(data[0]), "x", len(data))
# find the number of empty values in each column
empty_counter = []
for row in data:
    counter = 0
    for element in row:
        if element == "":
            counter += 1
    empty_counter.append(counter)

print("Number of empty values in each column:")
for i in range(len(titles)):
    if empty_counter[i] <100:
        print(titles[i], ":", empty_counter[i])
# # plot data[10] with vertical axes ranging from -50 to 150 in increments of 10
# print(counter)
# print(type(data[index][0]))
# plt.plot(data[index])
# plt.yticks(range(-1, 9, 2))
# plt.show()

titles_to_csv = ["DATE", "PRCP", "TMAX", "TMIN"]
# put these titles into a csv file
with open('Weather_data_Fresno_cleaned.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(titles_to_csv)
    for i in range(len(data[0])):
        writer.writerow([data[titles.index(titles_to_csv[0])][i], 
                         data[titles.index(titles_to_csv[1])][i], 
                         data[titles.index(titles_to_csv[2])][i], 
                         data[titles.index(titles_to_csv[3])][i]])