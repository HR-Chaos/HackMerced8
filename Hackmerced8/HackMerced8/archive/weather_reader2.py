import requests
import pandas as pd

# Set the API endpoint and parameters
url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data'
params = {
    'datasetid': 'GHCND',
    'locationid': 'FIPS:06019',  # San Joaquin Valley FIPS code
    'startdate': '1973-03-04',  # Start date of data
    'enddate': '2023-03-04',    # End date of data (today)
    'units': 'standard',         # Units in Fahrenheit and inches
    'limit': 1000,               # Max number of records to return
    'offset': 0,                 # Starting record offset
}

# Set the headers and API token
headers = {'token': 'uJCInoezrJnubCwEPdkREEPttPpZfYMe'}

# Send the API request and get the response
response = requests.get(url, headers=headers, params=params)
data = response.json()['results']

# Create a pandas DataFrame from the data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('san_joaquin_weather.csv', index=False)