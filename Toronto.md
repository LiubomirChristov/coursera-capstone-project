Downloading and preparing data for Toronto 



```python
import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis

import requests # library to handle requests

from sklearn.cluster import KMeans

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import json # library to handle JSON files

from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

print('Libraries imported.')
```

    Solving environment: done
    
    
    ==> WARNING: A newer version of conda exists. <==
      current version: 4.5.11
      latest version: 4.7.12
    
    Please update conda by running
    
        $ conda update -n base -c defaults conda
    
    
    
    # All requested packages already installed.
    
    Libraries imported.



```python
pip install lxml
```

    Requirement already satisfied: lxml in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (4.4.1)
    Note: you may need to restart the kernel to use updated packages.


Scrape the following Wikipedia page


```python
url = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
postcode_M_url = requests.get(url)
postcode_M_url
```




    <Response [200]>




```python
postcode_data = pd.read_html(postcode_M_url.text)
```


```python
len(postcode_data), type(postcode_data)
```




    (3, list)




```python
tbl1=postcode_data[0]
tbl1=tbl1.dropna() #drop the empty rows 
```


```python
tbl1.replace("Not assigned", np.nan, inplace = True)
tbl1.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postcode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1A</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M2A</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
    </tr>
  </tbody>
</table>
</div>




```python
# simply drop whole row with NaN in "Borough" column
tbl1.dropna(subset=["Borough"], axis=0, inplace=True)

# reset index, because we droped rows
tbl1.reset_index(drop=True, inplace=True)
```


```python
# If a cell has a borough but a Not assigned neighborhood, then the neighborhood will be the same as the borough
tbl1["Neighbourhood"].replace(np.nan, tbl1["Borough"], inplace=True)
```


```python
tbl1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postcode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Heights</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>205</th>
      <td>M8Z</td>
      <td>Etobicoke</td>
      <td>Kingsway Park South West</td>
    </tr>
    <tr>
      <th>206</th>
      <td>M8Z</td>
      <td>Etobicoke</td>
      <td>Mimico NW</td>
    </tr>
    <tr>
      <th>207</th>
      <td>M8Z</td>
      <td>Etobicoke</td>
      <td>The Queensway West</td>
    </tr>
    <tr>
      <th>208</th>
      <td>M8Z</td>
      <td>Etobicoke</td>
      <td>Royal York South West</td>
    </tr>
    <tr>
      <th>209</th>
      <td>M8Z</td>
      <td>Etobicoke</td>
      <td>South of Bloor</td>
    </tr>
  </tbody>
</table>
<p>210 rows × 3 columns</p>
</div>




```python
tb3=tbl1.sort_values(by=['Postcode'])
tb3.reset_index(drop=True, inplace=True)
tb3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postcode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Port Union</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Rouge Hill</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>205</th>
      <td>M9V</td>
      <td>Etobicoke</td>
      <td>Mount Olive</td>
    </tr>
    <tr>
      <th>206</th>
      <td>M9V</td>
      <td>Etobicoke</td>
      <td>South Steeles</td>
    </tr>
    <tr>
      <th>207</th>
      <td>M9V</td>
      <td>Etobicoke</td>
      <td>Thistletown</td>
    </tr>
    <tr>
      <th>208</th>
      <td>M9V</td>
      <td>Etobicoke</td>
      <td>Silverstone</td>
    </tr>
    <tr>
      <th>209</th>
      <td>M9W</td>
      <td>Etobicoke</td>
      <td>Northwest</td>
    </tr>
  </tbody>
</table>
<p>210 rows × 3 columns</p>
</div>




```python
# More than one neighborhood can exist in one postal code area.
for i in range(len(tb3)-1):
      if tb3.loc[i,"Postcode"]== tb3.loc[i+1, "Postcode"]:
        tb3.loc[i+1,"Neighbourhood"] = tb3.loc[i+1,"Neighbourhood"] + ", " + tb3.loc[i,"Neighbourhood"]
        tb3.drop(i, inplace=True)
print(tb3)
   
```

        Postcode      Borough                                      Neighbourhood
    1        M1B  Scarborough                                     Malvern, Rouge
    4        M1C  Scarborough             Highland Creek, Rouge Hill, Port Union
    7        M1E  Scarborough                  West Hill, Morningside, Guildwood
    8        M1G  Scarborough                                             Woburn
    9        M1H  Scarborough                                          Cedarbrae
    ..       ...          ...                                                ...
    195      M9N         York                                             Weston
    196      M9P    Etobicoke                                          Westmount
    200      M9R    Etobicoke  Martin Grove Gardens, St. Phillips, Kingsview ...
    208      M9V    Etobicoke  Silverstone, Thistletown, South Steeles, Mount...
    209      M9W    Etobicoke                                          Northwest
    
    [103 rows x 3 columns]



```python
tb3.reset_index(drop=True, inplace=True)
```


```python
tb3.tail(40)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postcode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>63</th>
      <td>M5N</td>
      <td>Central Toronto</td>
      <td>Roselawn</td>
    </tr>
    <tr>
      <th>64</th>
      <td>M5P</td>
      <td>Central Toronto</td>
      <td>Forest Hill West, Forest Hill North</td>
    </tr>
    <tr>
      <th>65</th>
      <td>M5R</td>
      <td>Central Toronto</td>
      <td>North Midtown, The Annex, Yorkville</td>
    </tr>
    <tr>
      <th>66</th>
      <td>M5S</td>
      <td>Downtown Toronto</td>
      <td>Harbord, University of Toronto</td>
    </tr>
    <tr>
      <th>67</th>
      <td>M5T</td>
      <td>Downtown Toronto</td>
      <td>Grange Park, Chinatown, Kensington Market</td>
    </tr>
    <tr>
      <th>68</th>
      <td>M5V</td>
      <td>Downtown Toronto</td>
      <td>Bathurst Quay, Harbourfront West, Island airpo...</td>
    </tr>
    <tr>
      <th>69</th>
      <td>M5W</td>
      <td>Downtown Toronto</td>
      <td>Stn A PO Boxes 25 The Esplanade</td>
    </tr>
    <tr>
      <th>70</th>
      <td>M5X</td>
      <td>Downtown Toronto</td>
      <td>First Canadian Place, Underground city</td>
    </tr>
    <tr>
      <th>71</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor, Lawrence Heights</td>
    </tr>
    <tr>
      <th>72</th>
      <td>M6B</td>
      <td>North York</td>
      <td>Glencairn</td>
    </tr>
    <tr>
      <th>73</th>
      <td>M6C</td>
      <td>York</td>
      <td>Humewood-Cedarvale</td>
    </tr>
    <tr>
      <th>74</th>
      <td>M6E</td>
      <td>York</td>
      <td>Caledonia-Fairbanks</td>
    </tr>
    <tr>
      <th>75</th>
      <td>M6G</td>
      <td>Downtown Toronto</td>
      <td>Christie</td>
    </tr>
    <tr>
      <th>76</th>
      <td>M6H</td>
      <td>West Toronto</td>
      <td>Dovercourt Village, Dufferin</td>
    </tr>
    <tr>
      <th>77</th>
      <td>M6J</td>
      <td>West Toronto</td>
      <td>Trinity, Little Portugal</td>
    </tr>
    <tr>
      <th>78</th>
      <td>M6K</td>
      <td>West Toronto</td>
      <td>Parkdale Village, Brockton, Exhibition Place</td>
    </tr>
    <tr>
      <th>79</th>
      <td>M6L</td>
      <td>North York</td>
      <td>Upwood Park, North Park, Downsview</td>
    </tr>
    <tr>
      <th>80</th>
      <td>M6M</td>
      <td>York</td>
      <td>Del Ray, Keelesdale, Mount Dennis, Silverthorn</td>
    </tr>
    <tr>
      <th>81</th>
      <td>M6N</td>
      <td>York</td>
      <td>Runnymede, The Junction North</td>
    </tr>
    <tr>
      <th>82</th>
      <td>M6P</td>
      <td>West Toronto</td>
      <td>The Junction South, High Park</td>
    </tr>
    <tr>
      <th>83</th>
      <td>M6R</td>
      <td>West Toronto</td>
      <td>Parkdale, Roncesvalles</td>
    </tr>
    <tr>
      <th>84</th>
      <td>M6S</td>
      <td>West Toronto</td>
      <td>Swansea, Runnymede</td>
    </tr>
    <tr>
      <th>85</th>
      <td>M7A</td>
      <td>Queen's Park</td>
      <td>Queen's Park</td>
    </tr>
    <tr>
      <th>86</th>
      <td>M7R</td>
      <td>Mississauga</td>
      <td>Canada Post Gateway Processing Centre</td>
    </tr>
    <tr>
      <th>87</th>
      <td>M7Y</td>
      <td>East Toronto</td>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
    </tr>
    <tr>
      <th>88</th>
      <td>M8V</td>
      <td>Etobicoke</td>
      <td>Humber Bay Shores, New Toronto, Mimico South</td>
    </tr>
    <tr>
      <th>89</th>
      <td>M8W</td>
      <td>Etobicoke</td>
      <td>Alderwood, Long Branch</td>
    </tr>
    <tr>
      <th>90</th>
      <td>M8X</td>
      <td>Etobicoke</td>
      <td>The Kingsway, Montgomery Road, Old Mill North</td>
    </tr>
    <tr>
      <th>91</th>
      <td>M8Y</td>
      <td>Etobicoke</td>
      <td>Kingsway Park South East, Mimico NE, Sunnylea,...</td>
    </tr>
    <tr>
      <th>92</th>
      <td>M8Z</td>
      <td>Etobicoke</td>
      <td>Royal York South West, South of Bloor, The Que...</td>
    </tr>
    <tr>
      <th>93</th>
      <td>M9A</td>
      <td>Queen's Park</td>
      <td>Queen's Park</td>
    </tr>
    <tr>
      <th>94</th>
      <td>M9B</td>
      <td>Etobicoke</td>
      <td>Islington, Martin Grove, Princess Gardens, Wes...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>M9C</td>
      <td>Etobicoke</td>
      <td>Bloordale Gardens, Eringate, Markland Wood, Ol...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>M9L</td>
      <td>North York</td>
      <td>Humber Summit</td>
    </tr>
    <tr>
      <th>97</th>
      <td>M9M</td>
      <td>North York</td>
      <td>Emery, Humberlea</td>
    </tr>
    <tr>
      <th>98</th>
      <td>M9N</td>
      <td>York</td>
      <td>Weston</td>
    </tr>
    <tr>
      <th>99</th>
      <td>M9P</td>
      <td>Etobicoke</td>
      <td>Westmount</td>
    </tr>
    <tr>
      <th>100</th>
      <td>M9R</td>
      <td>Etobicoke</td>
      <td>Martin Grove Gardens, St. Phillips, Kingsview ...</td>
    </tr>
    <tr>
      <th>101</th>
      <td>M9V</td>
      <td>Etobicoke</td>
      <td>Silverstone, Thistletown, South Steeles, Mount...</td>
    </tr>
    <tr>
      <th>102</th>
      <td>M9W</td>
      <td>Etobicoke</td>
      <td>Northwest</td>
    </tr>
  </tbody>
</table>
</div>



Reading geographical coordinates of each postal code


```python
path='http://cocl.us/Geospatial_data'
G_data = pd.read_csv(path)
```


```python
tb5=pd.merge(tb3, G_data, left_on='Postcode', right_on='Postal Code', how='left').drop('Postal Code', axis=1) # add coordinates to our dataframe
```


```python
tb5
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postcode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>West Hill, Morningside, Guildwood</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>98</th>
      <td>M9N</td>
      <td>York</td>
      <td>Weston</td>
      <td>43.706876</td>
      <td>-79.518188</td>
    </tr>
    <tr>
      <th>99</th>
      <td>M9P</td>
      <td>Etobicoke</td>
      <td>Westmount</td>
      <td>43.696319</td>
      <td>-79.532242</td>
    </tr>
    <tr>
      <th>100</th>
      <td>M9R</td>
      <td>Etobicoke</td>
      <td>Martin Grove Gardens, St. Phillips, Kingsview ...</td>
      <td>43.688905</td>
      <td>-79.554724</td>
    </tr>
    <tr>
      <th>101</th>
      <td>M9V</td>
      <td>Etobicoke</td>
      <td>Silverstone, Thistletown, South Steeles, Mount...</td>
      <td>43.739416</td>
      <td>-79.588437</td>
    </tr>
    <tr>
      <th>102</th>
      <td>M9W</td>
      <td>Etobicoke</td>
      <td>Northwest</td>
      <td>43.706748</td>
      <td>-79.594054</td>
    </tr>
  </tbody>
</table>
<p>103 rows × 5 columns</p>
</div>




```python
# selecting boroughs that contain the word Toronto 
toronto_df = tb5[tb5["Borough"].str.contains("Toronto")]
toronto_df.reset_index(drop=True, inplace=True)
```

#### Create a map of Toronto with neighborhoods superimposed on top.


```python
address = 'Toronto, ON'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))
```

    The geograpical coordinate of Toronto are 43.653963, -79.387207.



```python
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=12)

# add markers to map
for lat, lng, borough, neighborhood in zip(toronto_df['Latitude'], toronto_df['Longitude'], toronto_df['Borough'], toronto_df['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2I1NWU5YzFkODg3YzRkMzZhYTcyY2M1ZmJhZWNjMTQzIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9iNTVlOWMxZDg4N2M0ZDM2YWE3MmNjNWZiYWVjYzE0MyA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9iNTVlOWMxZDg4N2M0ZDM2YWE3MmNjNWZiYWVjYzE0MycsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUzOTYzLC03OS4zODcyMDddLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgem9vbTogMTIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYXhCb3VuZHM6IGJvdW5kcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxheWVyczogW10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3b3JsZENvcHlKdW1wOiBmYWxzZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSk7CiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyX2YxNWM0ZDlmNWNjMDRjMmFiNTAxYzk0MDk2Y2JiYjA5ID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgICAgICAgICAgewogICJhdHRyaWJ1dGlvbiI6IG51bGwsCiAgImRldGVjdFJldGluYSI6IGZhbHNlLAogICJtYXhab29tIjogMTgsCiAgIm1pblpvb20iOiAxLAogICJub1dyYXAiOiBmYWxzZSwKICAic3ViZG9tYWlucyI6ICJhYmMiCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I1NWU5YzFkODg3YzRkMzZhYTcyY2M1ZmJhZWNjMTQzKTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iZGVkZDkwNzg3MzE0MmM1OWEyZTkxMTc0ZjdmMTVlZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3NjM1NzM5OTk5OTk5LC03OS4yOTMwMzEyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I1NWU5YzFkODg3YzRkMzZhYTcyY2M1ZmJhZWNjMTQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RlNTU0ZDRjZmIzZDQzMDZhOWU5YTFmZjQwZjdmZTM3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk3YmMwMDc3ODI5ZjQ2NDc5MDQ4ZDA2OWRjY2E1ZGIwID0gJCgnPGRpdiBpZD0iaHRtbF85N2JjMDA3NzgyOWY0NjQ3OTA0OGQwNjlkY2NhNWRiMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEJlYWNoZXMsIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGU1NTRkNGNmYjNkNDMwNmE5ZTlhMWZmNDBmN2ZlMzcuc2V0Q29udGVudChodG1sXzk3YmMwMDc3ODI5ZjQ2NDc5MDQ4ZDA2OWRjY2E1ZGIwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2JkZWRkOTA3ODczMTQyYzU5YTJlOTExNzRmN2YxNWVmLmJpbmRQb3B1cChwb3B1cF9kZTU1NGQ0Y2ZiM2Q0MzA2YTllOWExZmY0MGY3ZmUzNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lYzA1OTk4YzU5NTg0ODgyOGNmNmQxZTRhYzllZmRmOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU1NzEsLTc5LjM1MjE4OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNTVlOWMxZDg4N2M0ZDM2YWE3MmNjNWZiYWVjYzE0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83Yjg5M2M5N2MzMmM0NWI4YTZjZmVhY2I1NTJhNTEwMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80MjA3NzhiYmQwNjk0OTFjYmM4OWRjNjExNmMwZTYxYyA9ICQoJzxkaXYgaWQ9Imh0bWxfNDIwNzc4YmJkMDY5NDkxY2JjODlkYzYxMTZjMGU2MWMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBEYW5mb3J0aCBXZXN0LCBSaXZlcmRhbGUsIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfN2I4OTNjOTdjMzJjNDViOGE2Y2ZlYWNiNTUyYTUxMDMuc2V0Q29udGVudChodG1sXzQyMDc3OGJiZDA2OTQ5MWNiYzg5ZGM2MTE2YzBlNjFjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2VjMDU5OThjNTk1ODQ4ODI4Y2Y2ZDFlNGFjOWVmZGY5LmJpbmRQb3B1cChwb3B1cF83Yjg5M2M5N2MzMmM0NWI4YTZjZmVhY2I1NTJhNTEwMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81YzNkZTU4ZGZjM2E0ZTJiOTYwMWUxZmY5MjRjNzQyNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2ODk5ODUsLTc5LjMxNTU3MTU5OTk5OTk4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I1NWU5YzFkODg3YzRkMzZhYTcyY2M1ZmJhZWNjMTQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RlNmVmZGUzYzhiNDRmZjJhYTdhMGJhNjBjZGUzMzAzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I0YTVjNzU3MzQwYTRkNThiNWQzOWFhZTRjNGIzNTcxID0gJCgnPGRpdiBpZD0iaHRtbF9iNGE1Yzc1NzM0MGE0ZDU4YjVkMzlhYWU0YzRiMzU3MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SW5kaWEgQmF6YWFyLCBUaGUgQmVhY2hlcyBXZXN0LCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RlNmVmZGUzYzhiNDRmZjJhYTdhMGJhNjBjZGUzMzAzLnNldENvbnRlbnQoaHRtbF9iNGE1Yzc1NzM0MGE0ZDU4YjVkMzlhYWU0YzRiMzU3MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81YzNkZTU4ZGZjM2E0ZTJiOTYwMWUxZmY5MjRjNzQyNi5iaW5kUG9wdXAocG9wdXBfZGU2ZWZkZTNjOGI0NGZmMmFhN2EwYmE2MGNkZTMzMDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmI3NTQ5ZDExYTFkNGNkYThhMTBkMzRkNjBlZGMzMTYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTk1MjU1LC03OS4zNDA5MjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTNhMDYwMWNiMGI5NDQ3NzhjMThhM2U4NjNhZTUwYjkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTU2NDZkZThlYjBhNDkxMWJiMDZkZmFhNmM5ZWFmNTIgPSAkKCc8ZGl2IGlkPSJodG1sXzk1NjQ2ZGU4ZWIwYTQ5MTFiYjA2ZGZhYTZjOWVhZjUyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdHVkaW8gRGlzdHJpY3QsIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTNhMDYwMWNiMGI5NDQ3NzhjMThhM2U4NjNhZTUwYjkuc2V0Q29udGVudChodG1sXzk1NjQ2ZGU4ZWIwYTQ5MTFiYjA2ZGZhYTZjOWVhZjUyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZiNzU0OWQxMWExZDRjZGE4YTEwZDM0ZDYwZWRjMzE2LmJpbmRQb3B1cChwb3B1cF9hM2EwNjAxY2IwYjk0NDc3OGMxOGEzZTg2M2FlNTBiOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xOTFjN2ZiYjMzZTk0ZmZiYmY5NTI0MGUxMzI1YzAxMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyODAyMDUsLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMmQ5MTBmY2ExM2U5NDcxY2E1Mjg3ZGY2MDYxZWQxYzEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDhiZjYyOTY1NjhlNDkzNWFlMTgzMzA1ZGE5MjliNDEgPSAkKCc8ZGl2IGlkPSJodG1sXzQ4YmY2Mjk2NTY4ZTQ5MzVhZTE4MzMwNWRhOTI5YjQxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MYXdyZW5jZSBQYXJrLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzJkOTEwZmNhMTNlOTQ3MWNhNTI4N2RmNjA2MWVkMWMxLnNldENvbnRlbnQoaHRtbF80OGJmNjI5NjU2OGU0OTM1YWUxODMzMDVkYTkyOWI0MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xOTFjN2ZiYjMzZTk0ZmZiYmY5NTI0MGUxMzI1YzAxMS5iaW5kUG9wdXAocG9wdXBfMmQ5MTBmY2ExM2U5NDcxY2E1Mjg3ZGY2MDYxZWQxYzEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjczZGY5ODc1NDc3NGZjZjg5NTdhNWM3YWU4MTFmYTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTI3NTExLC03OS4zOTAxOTc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I1NWU5YzFkODg3YzRkMzZhYTcyY2M1ZmJhZWNjMTQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2VhYjZiOTZjOGVkMjQ2YTJhNzllNjM2M2VjMjUxMWRlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2JhMzgxNDZjMWQ5NjRlM2JiZjAyZjgyYTJkZThlNmRkID0gJCgnPGRpdiBpZD0iaHRtbF9iYTM4MTQ2YzFkOTY0ZTNiYmYwMmY4MmEyZGU4ZTZkZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGF2aXN2aWxsZSBOb3J0aCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lYWI2Yjk2YzhlZDI0NmEyYTc5ZTYzNjNlYzI1MTFkZS5zZXRDb250ZW50KGh0bWxfYmEzODE0NmMxZDk2NGUzYmJmMDJmODJhMmRlOGU2ZGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjczZGY5ODc1NDc3NGZjZjg5NTdhNWM3YWU4MTFmYTguYmluZFBvcHVwKHBvcHVwX2VhYjZiOTZjOGVkMjQ2YTJhNzllNjM2M2VjMjUxMWRlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzA5MDhjYjE1ZmZkYzQ3YmNhMmI3ODA3MWM5ZThmMjhkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzE1MzgzNCwtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMThkY2YzNjE5M2M1NGExNzgwODgyNWI3MmU2ZGUyODMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNmViMDRlZjkyMWIwNGYyM2I3YmU2ZWNmNWVhNDIwODkgPSAkKCc8ZGl2IGlkPSJodG1sXzZlYjA0ZWY5MjFiMDRmMjNiN2JlNmVjZjVlYTQyMDg5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aCBUb3JvbnRvIFdlc3QsIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMThkY2YzNjE5M2M1NGExNzgwODgyNWI3MmU2ZGUyODMuc2V0Q29udGVudChodG1sXzZlYjA0ZWY5MjFiMDRmMjNiN2JlNmVjZjVlYTQyMDg5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzA5MDhjYjE1ZmZkYzQ3YmNhMmI3ODA3MWM5ZThmMjhkLmJpbmRQb3B1cChwb3B1cF8xOGRjZjM2MTkzYzU0YTE3ODA4ODI1YjcyZTZkZTI4Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wN2FhNTI0NmU1MTQ0OGMyOWE2YzRhMTVmYTYzNWY5ZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNDMyNDQsLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzQxMjUxNzFkMjdiNDJlNjhlNjA3ZmEzOWFjYjg1OWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTc1MGUxMmFmODZiNDg1NWJkN2ZiZjBkMGVjMDg1MjkgPSAkKCc8ZGl2IGlkPSJodG1sXzU3NTBlMTJhZjg2YjQ4NTViZDdmYmYwZDBlYzA4NTI5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EYXZpc3ZpbGxlLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzc0MTI1MTcxZDI3YjQyZTY4ZTYwN2ZhMzlhY2I4NTliLnNldENvbnRlbnQoaHRtbF81NzUwZTEyYWY4NmI0ODU1YmQ3ZmJmMGQwZWMwODUyOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wN2FhNTI0NmU1MTQ0OGMyOWE2YzRhMTVmYTYzNWY5Zi5iaW5kUG9wdXAocG9wdXBfNzQxMjUxNzFkMjdiNDJlNjhlNjA3ZmEzOWFjYjg1OWIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2NjZmI0NTVhOWI5NDAyYTllZWIwMDk3M2Y3ZDZkMmIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODk1NzQzLC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNTVlOWMxZDg4N2M0ZDM2YWE3MmNjNWZiYWVjYzE0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jY2Y0MjdkNWQ0MDc0Yzk5YTJkMzBlYjQ0Njk4NDNmYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85YmU2ODRhNTg4MTc0NGQ4YjE2ZGE1MTA3YjJkNjQzZSA9ICQoJzxkaXYgaWQ9Imh0bWxfOWJlNjg0YTU4ODE3NDRkOGIxNmRhNTEwN2IyZDY0M2UiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1vb3JlIFBhcmssIFN1bW1lcmhpbGwgRWFzdCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jY2Y0MjdkNWQ0MDc0Yzk5YTJkMzBlYjQ0Njk4NDNmYy5zZXRDb250ZW50KGh0bWxfOWJlNjg0YTU4ODE3NDRkOGIxNmRhNTEwN2IyZDY0M2UpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2NjZmI0NTVhOWI5NDAyYTllZWIwMDk3M2Y3ZDZkMmIuYmluZFBvcHVwKHBvcHVwX2NjZjQyN2Q1ZDQwNzRjOTlhMmQzMGViNDQ2OTg0M2ZjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNmYzQ5YWNhZTYwNDQ3NWE4NDU3OWNlM2I5Yzg4NzE4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg2NDEyMjk5OTk5OTksLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjdmZTEyNmViMWMzNDE2ZWE2ZTAzNTIzYThmMGUwYjUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmI0MGU5Y2JjMTUxNGE3MjhkNjZlYWRlNzhjNjg0ZTEgPSAkKCc8ZGl2IGlkPSJodG1sX2JiNDBlOWNiYzE1MTRhNzI4ZDY2ZWFkZTc4YzY4NGUxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EZWVyIFBhcmssIEZvcmVzdCBIaWxsIFNFLCBSYXRobmVsbHksIFN1bW1lcmhpbGwgV2VzdCwgU291dGggSGlsbCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iN2ZlMTI2ZWIxYzM0MTZlYTZlMDM1MjNhOGYwZTBiNS5zZXRDb250ZW50KGh0bWxfYmI0MGU5Y2JjMTUxNGE3MjhkNjZlYWRlNzhjNjg0ZTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2ZjNDlhY2FlNjA0NDc1YTg0NTc5Y2UzYjljODg3MTguYmluZFBvcHVwKHBvcHVwX2I3ZmUxMjZlYjFjMzQxNmVhNmUwMzUyM2E4ZjBlMGI1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M4MDFkZWRhMjFhOTQ0MDViMTljMTA4MTY4MjE2M2UwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc5NTYyNiwtNzkuMzc3NTI5NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjI2NjZiNzliNTYxNGYyMjlmOThjZmVjMzNmMDYyNmYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjZlY2U5MDFjNzFlNGU4N2I5NTU5MTA5ZjViMWFjZDMgPSAkKCc8ZGl2IGlkPSJodG1sXzY2ZWNlOTAxYzcxZTRlODdiOTU1OTEwOWY1YjFhY2QzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3NlZGFsZSwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjI2NjZiNzliNTYxNGYyMjlmOThjZmVjMzNmMDYyNmYuc2V0Q29udGVudChodG1sXzY2ZWNlOTAxYzcxZTRlODdiOTU1OTEwOWY1YjFhY2QzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M4MDFkZWRhMjFhOTQ0MDViMTljMTA4MTY4MjE2M2UwLmJpbmRQb3B1cChwb3B1cF9mMjY2NmI3OWI1NjE0ZjIyOWY5OGNmZWMzM2YwNjI2Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mMzk2ZTVhOThkOGI0ZGU5YjNjNGYzNmNmNjViY2FlOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzk2NywtNzkuMzY3Njc1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNTVlOWMxZDg4N2M0ZDM2YWE3MmNjNWZiYWVjYzE0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYTFhZGI3NWJlOWU0MmIzOTkxOTk3NDhkNjcxYTk5MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yZTU1NDQ1OTNlMzU0NWIzYjM1NjNjM2NlZGU3NTY5MiA9ICQoJzxkaXYgaWQ9Imh0bWxfMmU1NTQ0NTkzZTM1NDViM2IzNTYzYzNjZWRlNzU2OTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNhYmJhZ2V0b3duLCBTdC4gSmFtZXMgVG93biwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2ExYWRiNzViZTllNDJiMzk5MTk5NzQ4ZDY3MWE5OTEuc2V0Q29udGVudChodG1sXzJlNTU0NDU5M2UzNTQ1YjNiMzU2M2MzY2VkZTc1NjkyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2YzOTZlNWE5OGQ4YjRkZTliM2M0ZjM2Y2Y2NWJjYWU4LmJpbmRQb3B1cChwb3B1cF9jYTFhZGI3NWJlOWU0MmIzOTkxOTk3NDhkNjcxYTk5MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84MmZkMWE0ODk1YjU0ZmZiYTEyZTA5MTczNzU0YzE5YSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2NTg1OTksLTc5LjM4MzE1OTkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I1NWU5YzFkODg3YzRkMzZhYTcyY2M1ZmJhZWNjMTQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzdhZWViNjA1YzJmYTQ0NDY5MDdhYjZmNDAxMDU0NjM1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E3YWExNWJjNDgzZDQ4MGU5OGFmNGYzYWIxMTgzMWY1ID0gJCgnPGRpdiBpZD0iaHRtbF9hN2FhMTViYzQ4M2Q0ODBlOThhZjRmM2FiMTE4MzFmNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2h1cmNoIGFuZCBXZWxsZXNsZXksIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzdhZWViNjA1YzJmYTQ0NDY5MDdhYjZmNDAxMDU0NjM1LnNldENvbnRlbnQoaHRtbF9hN2FhMTViYzQ4M2Q0ODBlOThhZjRmM2FiMTE4MzFmNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84MmZkMWE0ODk1YjU0ZmZiYTEyZTA5MTczNzU0YzE5YS5iaW5kUG9wdXAocG9wdXBfN2FlZWI2MDVjMmZhNDQ0NjkwN2FiNmY0MDEwNTQ2MzUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNGI3NmI0MWFkNGE5NGY3OTljYTljODJmZjcxYjk0MjAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTQyNTk5LC03OS4zNjA2MzU5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I1NWU5YzFkODg3YzRkMzZhYTcyY2M1ZmJhZWNjMTQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzcwZjU2YzE2OGU4MzRlY2NiZmUxYjMzYzg5M2M1YjY4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzAyNDJhYmU0MzE0NTQ4ZTE4YmJlZDY1MzRjM2RmYjZmID0gJCgnPGRpdiBpZD0iaHRtbF8wMjQyYWJlNDMxNDU0OGUxOGJiZWQ2NTM0YzNkZmI2ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm91cmZyb250LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83MGY1NmMxNjhlODM0ZWNjYmZlMWIzM2M4OTNjNWI2OC5zZXRDb250ZW50KGh0bWxfMDI0MmFiZTQzMTQ1NDhlMThiYmVkNjUzNGMzZGZiNmYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGI3NmI0MWFkNGE5NGY3OTljYTljODJmZjcxYjk0MjAuYmluZFBvcHVwKHBvcHVwXzcwZjU2YzE2OGU4MzRlY2NiZmUxYjMzYzg5M2M1YjY4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFiNTQ1M2UxODQyYjQ4ZTY4YzRiYjI2YzU5YjFmNWRmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3MTYxOCwtNzkuMzc4OTM3MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGY5MjdjMTY4YTMyNDdmMTljODA4ZGNjNTFhM2U0ODQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDZjMTg3M2ViN2ZhNDUyNDkyNmY0OWI3YjRhYjcyOTIgPSAkKCc8ZGl2IGlkPSJodG1sXzA2YzE4NzNlYjdmYTQ1MjQ5MjZmNDliN2I0YWI3MjkyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SeWVyc29uLCBHYXJkZW4gRGlzdHJpY3QsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRmOTI3YzE2OGEzMjQ3ZjE5YzgwOGRjYzUxYTNlNDg0LnNldENvbnRlbnQoaHRtbF8wNmMxODczZWI3ZmE0NTI0OTI2ZjQ5YjdiNGFiNzI5Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xYjU0NTNlMTg0MmI0OGU2OGM0YmIyNmM1OWIxZjVkZi5iaW5kUG9wdXAocG9wdXBfNGY5MjdjMTY4YTMyNDdmMTljODA4ZGNjNTFhM2U0ODQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTJkZDU3YWM5NTQ2NDg2N2I2MzUzNmE1ODRkNWJiY2EgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE0OTM5LC03OS4zNzU0MTc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I1NWU5YzFkODg3YzRkMzZhYTcyY2M1ZmJhZWNjMTQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzgyZjczZDQ4NTdjNzQ0NWU5YWJkMjYxYTNjOWI3ZThhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZlNzIyMDIxYmVhMDQxNzE4YTlkYWY2MjM3ZGU1NDk5ID0gJCgnPGRpdiBpZD0iaHRtbF9mZTcyMjAyMWJlYTA0MTcxOGE5ZGFmNjIzN2RlNTQ5OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3QuIEphbWVzIFRvd24sIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzgyZjczZDQ4NTdjNzQ0NWU5YWJkMjYxYTNjOWI3ZThhLnNldENvbnRlbnQoaHRtbF9mZTcyMjAyMWJlYTA0MTcxOGE5ZGFmNjIzN2RlNTQ5OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xMmRkNTdhYzk1NDY0ODY3YjYzNTM2YTU4NGQ1YmJjYS5iaW5kUG9wdXAocG9wdXBfODJmNzNkNDg1N2M3NDQ1ZTlhYmQyNjFhM2M5YjdlOGEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWE2NGNiMTJkMDU2NDFlZTgyZGJiZDIxNzFiMjlmNWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDQ3NzA3OTk5OTk5OTYsLTc5LjM3MzMwNjRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmEyYmFlMGU2MGZlNDllODhlY2FmY2JkNzgxNzk5ZjMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTY1NDIyMTMzMTdmNDU2NzkzZWZhYzEyNDcyNmMwYzUgPSAkKCc8ZGl2IGlkPSJodG1sXzk2NTQyMjEzMzE3ZjQ1Njc5M2VmYWMxMjQ3MjZjMGM1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CZXJjenkgUGFyaywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmEyYmFlMGU2MGZlNDllODhlY2FmY2JkNzgxNzk5ZjMuc2V0Q29udGVudChodG1sXzk2NTQyMjEzMzE3ZjQ1Njc5M2VmYWMxMjQ3MjZjMGM1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzVhNjRjYjEyZDA1NjQxZWU4MmRiYmQyMTcxYjI5ZjVmLmJpbmRQb3B1cChwb3B1cF82YTJiYWUwZTYwZmU0OWU4OGVjYWZjYmQ3ODE3OTlmMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wNDVkMmU2YTg4YjE0MTJiOTY1YTcxMmFmYzM0MTVmMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1Nzk1MjQsLTc5LjM4NzM4MjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOGVmNDgyZTNjOGQxNDYyNGFjM2U1NzNlYTNlYzI3NDYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzBlNTRhNGEwZWEzNGUyYmIxN2Q1NTJkZWI4MzYyMDEgPSAkKCc8ZGl2IGlkPSJodG1sXzcwZTU0YTRhMGVhMzRlMmJiMTdkNTUyZGViODM2MjAxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIEJheSBTdHJlZXQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhlZjQ4MmUzYzhkMTQ2MjRhYzNlNTczZWEzZWMyNzQ2LnNldENvbnRlbnQoaHRtbF83MGU1NGE0YTBlYTM0ZTJiYjE3ZDU1MmRlYjgzNjIwMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wNDVkMmU2YTg4YjE0MTJiOTY1YTcxMmFmYzM0MTVmMC5iaW5kUG9wdXAocG9wdXBfOGVmNDgyZTNjOGQxNDYyNGFjM2U1NzNlYTNlYzI3NDYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYmQ2YmEwNmYyZDIxNDM1N2EwMmY0MTIyN2NkMmJhMDAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTA1NzEyMDAwMDAwMSwtNzkuMzg0NTY3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNTVlOWMxZDg4N2M0ZDM2YWE3MmNjNWZiYWVjYzE0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lZTkzZTljOTdjZmY0MmFhYTgyYWEwMmNmM2ZjY2E4MCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kOTc4OTI5OGIxZDY0NzgxODFiNWEyNWU5NWQxMDEzOCA9ICQoJzxkaXYgaWQ9Imh0bWxfZDk3ODkyOThiMWQ2NDc4MTgxYjVhMjVlOTVkMTAxMzgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPktpbmcsIFJpY2htb25kLCBBZGVsYWlkZSwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZWU5M2U5Yzk3Y2ZmNDJhYWE4MmFhMDJjZjNmY2NhODAuc2V0Q29udGVudChodG1sX2Q5Nzg5Mjk4YjFkNjQ3ODE4MWI1YTI1ZTk1ZDEwMTM4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2JkNmJhMDZmMmQyMTQzNTdhMDJmNDEyMjdjZDJiYTAwLmJpbmRQb3B1cChwb3B1cF9lZTkzZTljOTdjZmY0MmFhYTgyYWEwMmNmM2ZjY2E4MCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84ODg4YWYwY2MxOGM0MzI2OWYxMTE3OGIxNzJjODA3YSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0MDgxNTcsLTc5LjM4MTc1MjI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I1NWU5YzFkODg3YzRkMzZhYTcyY2M1ZmJhZWNjMTQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzczNmE5ZGY2ZWEwYTQzYWJhYTlmN2Y2MDAwMTJhZWQzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzE4MDFlMDI0MDc2MjQ1MDhhNzhjMzg0YjRlOTNlNzJhID0gJCgnPGRpdiBpZD0iaHRtbF8xODAxZTAyNDA3NjI0NTA4YTc4YzM4NGI0ZTkzZTcyYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VG9yb250byBJc2xhbmRzLCBVbmlvbiBTdGF0aW9uLCBIYXJib3VyZnJvbnQgRWFzdCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzM2YTlkZjZlYTBhNDNhYmFhOWY3ZjYwMDAxMmFlZDMuc2V0Q29udGVudChodG1sXzE4MDFlMDI0MDc2MjQ1MDhhNzhjMzg0YjRlOTNlNzJhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg4ODhhZjBjYzE4YzQzMjY5ZjExMTc4YjE3MmM4MDdhLmJpbmRQb3B1cChwb3B1cF83MzZhOWRmNmVhMGE0M2FiYWE5ZjdmNjAwMDEyYWVkMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lMjlmZjE5NGI5ZTU0MmM0OWMwNDc0ZWZiOTY5YzZmOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzE3NjgsLTc5LjM4MTU3NjQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I1NWU5YzFkODg3YzRkMzZhYTcyY2M1ZmJhZWNjMTQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUzYTY2ZTVlYTM0MjQwYjViMDQ1ODk2MmVlMTUwODhiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzIzODZkOTlkNDM0YzQ5ZGE4MjljYWY3OWUwZWNhODZkID0gJCgnPGRpdiBpZD0iaHRtbF8yMzg2ZDk5ZDQzNGM0OWRhODI5Y2FmNzllMGVjYTg2ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VG9yb250byBEb21pbmlvbiBDZW50cmUsIERlc2lnbiBFeGNoYW5nZSwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTNhNjZlNWVhMzQyNDBiNWIwNDU4OTYyZWUxNTA4OGIuc2V0Q29udGVudChodG1sXzIzODZkOTlkNDM0YzQ5ZGE4MjljYWY3OWUwZWNhODZkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2UyOWZmMTk0YjllNTQyYzQ5YzA0NzRlZmI5NjljNmY5LmJpbmRQb3B1cChwb3B1cF81M2E2NmU1ZWEzNDI0MGI1YjA0NTg5NjJlZTE1MDg4Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wODJlOTJmNGM3NDQ0ZGJmOWE2ZjAwN2U4ZjI4MGM2ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODE5ODUsLTc5LjM3OTgxNjkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I1NWU5YzFkODg3YzRkMzZhYTcyY2M1ZmJhZWNjMTQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU2MDU1NDEwM2ZjMjQ3MWE4MDk2YzI3NGFiZTVjMTBjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzM5OWYxNWQxYmViZTQzMjlhMDEzN2UyYTU4NjY0Y2Q0ID0gJCgnPGRpdiBpZD0iaHRtbF8zOTlmMTVkMWJlYmU0MzI5YTAxMzdlMmE1ODY2NGNkNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VmljdG9yaWEgSG90ZWwsIENvbW1lcmNlIENvdXJ0LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81NjA1NTQxMDNmYzI0NzFhODA5NmMyNzRhYmU1YzEwYy5zZXRDb250ZW50KGh0bWxfMzk5ZjE1ZDFiZWJlNDMyOWEwMTM3ZTJhNTg2NjRjZDQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDgyZTkyZjRjNzQ0NGRiZjlhNmYwMDdlOGYyODBjNmQuYmluZFBvcHVwKHBvcHVwXzU2MDU1NDEwM2ZjMjQ3MWE4MDk2YzI3NGFiZTVjMTBjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2IwOGI2OGRkM2U4MzQxMzRiZDIyOGM3ZDllYmNiMWEwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzExNjk0OCwtNzkuNDE2OTM1NTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzRlYjNjMTgyODQ2NGZiMWJhOTVjZmQzOGE5NjgwYjggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzkxMWExNjEwODA5NGU3NmI3Y2EwYjc5ZTc2NjgyYjUgPSAkKCc8ZGl2IGlkPSJodG1sX2M5MTFhMTYxMDgwOTRlNzZiN2NhMGI3OWU3NjY4MmI1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3NlbGF3biwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zNGViM2MxODI4NDY0ZmIxYmE5NWNmZDM4YTk2ODBiOC5zZXRDb250ZW50KGh0bWxfYzkxMWExNjEwODA5NGU3NmI3Y2EwYjc5ZTc2NjgyYjUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjA4YjY4ZGQzZTgzNDEzNGJkMjI4YzdkOWViY2IxYTAuYmluZFBvcHVwKHBvcHVwXzM0ZWIzYzE4Mjg0NjRmYjFiYTk1Y2ZkMzhhOTY4MGI4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzkwZjNmZjliOGIxMzQ0MTZiYjA1OTNiNWRiYTY1MTBkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjk2OTQ3NiwtNzkuNDExMzA3MjAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWQyZDRlODA3NjY3NDZkYTk0OGQ5Mzk2MjBlNjNkYmYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTMxNWQxM2YzMDIwNGUxNGFjNGVmMjBjNTI3ZmVjMDMgPSAkKCc8ZGl2IGlkPSJodG1sXzkzMTVkMTNmMzAyMDRlMTRhYzRlZjIwYzUyN2ZlYzAzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Gb3Jlc3QgSGlsbCBXZXN0LCBGb3Jlc3QgSGlsbCBOb3J0aCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81ZDJkNGU4MDc2Njc0NmRhOTQ4ZDkzOTYyMGU2M2RiZi5zZXRDb250ZW50KGh0bWxfOTMxNWQxM2YzMDIwNGUxNGFjNGVmMjBjNTI3ZmVjMDMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTBmM2ZmOWI4YjEzNDQxNmJiMDU5M2I1ZGJhNjUxMGQuYmluZFBvcHVwKHBvcHVwXzVkMmQ0ZTgwNzY2NzQ2ZGE5NDhkOTM5NjIwZTYzZGJmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I2NmYxZDc1OTRkNzQ0ZWFiZGQ0YTIxZTQwZTQ0YTRhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjcyNzA5NywtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTcwZjAyOTY4YTBkNDE3YTgzZTY3NTBkMzZhZDNhODggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2JkYjM2YjAzNmE0NGY0OTgyMzg3YzExZGJiMGVmMjMgPSAkKCc8ZGl2IGlkPSJodG1sXzdiZGIzNmIwMzZhNDRmNDk4MjM4N2MxMWRiYjBlZjIzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aCBNaWR0b3duLCBUaGUgQW5uZXgsIFlvcmt2aWxsZSwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lNzBmMDI5NjhhMGQ0MTdhODNlNjc1MGQzNmFkM2E4OC5zZXRDb250ZW50KGh0bWxfN2JkYjM2YjAzNmE0NGY0OTgyMzg3YzExZGJiMGVmMjMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjY2ZjFkNzU5NGQ3NDRlYWJkZDRhMjFlNDBlNDRhNGEuYmluZFBvcHVwKHBvcHVwX2U3MGYwMjk2OGEwZDQxN2E4M2U2NzUwZDM2YWQzYTg4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZjMGQ5OGFkNjYxYzQwNTY4YWY3ODg4NzY4NDlkYTI3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNjk1NiwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNTVlOWMxZDg4N2M0ZDM2YWE3MmNjNWZiYWVjYzE0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMDQwN2M4ZmY3NDM0OTY0YWU1MDc1ZGVlNWE1Zjk3ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YzRiNzYwZGMzMDg0ZTI3YmY2N2U1NGUyM2E3YjEzNSA9ICQoJzxkaXYgaWQ9Imh0bWxfN2M0Yjc2MGRjMzA4NGUyN2JmNjdlNTRlMjNhN2IxMzUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhhcmJvcmQsIFVuaXZlcnNpdHkgb2YgVG9yb250bywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTA0MDdjOGZmNzQzNDk2NGFlNTA3NWRlZTVhNWY5N2Yuc2V0Q29udGVudChodG1sXzdjNGI3NjBkYzMwODRlMjdiZjY3ZTU0ZTIzYTdiMTM1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZjMGQ5OGFkNjYxYzQwNTY4YWY3ODg4NzY4NDlkYTI3LmJpbmRQb3B1cChwb3B1cF9lMDQwN2M4ZmY3NDM0OTY0YWU1MDc1ZGVlNWE1Zjk3Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lNGQ5YThmMmVlZTI0Mjk0YjRhOGYyMWM2MDQyNWI3MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MzIwNTcsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjNiNTAzM2FiNDczNDM4ZWJhZDcxODVkYTNmNDBiY2UgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZWJkMzZiOTIwODRkNDRhMWJhNTNhYzAyNGRmNmMyMDYgPSAkKCc8ZGl2IGlkPSJodG1sX2ViZDM2YjkyMDg0ZDQ0YTFiYTUzYWMwMjRkZjZjMjA2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HcmFuZ2UgUGFyaywgQ2hpbmF0b3duLCBLZW5zaW5ndG9uIE1hcmtldCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjNiNTAzM2FiNDczNDM4ZWJhZDcxODVkYTNmNDBiY2Uuc2V0Q29udGVudChodG1sX2ViZDM2YjkyMDg0ZDQ0YTFiYTUzYWMwMjRkZjZjMjA2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U0ZDlhOGYyZWVlMjQyOTRiNGE4ZjIxYzYwNDI1YjcyLmJpbmRQb3B1cChwb3B1cF82M2I1MDMzYWI0NzM0MzhlYmFkNzE4NWRhM2Y0MGJjZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yNDRmNGYyNDg0NTQ0ZGU5ODk1ZGM5MTUwMWFkMTZmNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWJhMmYxNGNiMjIxNGZkYTkxOTk4NDViOWNiY2M3MzUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWFjYzhjZDg4YmM1NGNkYTliYzBmZmNjZTYxMWU5OGMgPSAkKCc8ZGl2IGlkPSJodG1sXzVhY2M4Y2Q4OGJjNTRjZGE5YmMwZmZjY2U2MTFlOThjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CYXRodXJzdCBRdWF5LCBIYXJib3VyZnJvbnQgV2VzdCwgSXNsYW5kIGFpcnBvcnQsIENOIFRvd2VyLCBTb3V0aCBOaWFnYXJhLCBSYWlsd2F5IExhbmRzLCBLaW5nIGFuZCBTcGFkaW5hLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hYmEyZjE0Y2IyMjE0ZmRhOTE5OTg0NWI5Y2JjYzczNS5zZXRDb250ZW50KGh0bWxfNWFjYzhjZDg4YmM1NGNkYTliYzBmZmNjZTYxMWU5OGMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjQ0ZjRmMjQ4NDU0NGRlOTg5NWRjOTE1MDFhZDE2ZjYuYmluZFBvcHVwKHBvcHVwX2FiYTJmMTRjYjIyMTRmZGE5MTk5ODQ1YjljYmNjNzM1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJmNjNkOTZiNjNlYjQ2YzE5Zjk2YzBmOGFkMjY3NzEyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ2NDM1MiwtNzkuMzc0ODQ1OTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGYxOGQwNDczY2NmNDY1NGJhNDZhZTFiODRjOWY3MDIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTMzNmRlMjQ5NjUyNDEwNzk5ZjlhNGIxNWU1NDRjMzMgPSAkKCc8ZGl2IGlkPSJodG1sXzUzMzZkZTI0OTY1MjQxMDc5OWY5YTRiMTVlNTQ0YzMzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdG4gQSBQTyBCb3hlcyAyNSBUaGUgRXNwbGFuYWRlLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80ZjE4ZDA0NzNjY2Y0NjU0YmE0NmFlMWI4NGM5ZjcwMi5zZXRDb250ZW50KGh0bWxfNTMzNmRlMjQ5NjUyNDEwNzk5ZjlhNGIxNWU1NDRjMzMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmY2M2Q5NmI2M2ViNDZjMTlmOTZjMGY4YWQyNjc3MTIuYmluZFBvcHVwKHBvcHVwXzRmMThkMDQ3M2NjZjQ2NTRiYTQ2YWUxYjg0YzlmNzAyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2QwNWYwMWI4NDEwZTQxNDNiYTRlZDNkNjVjYjQxNDI3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4NDI5MiwtNzkuMzgyMjgwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNTVlOWMxZDg4N2M0ZDM2YWE3MmNjNWZiYWVjYzE0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zOWE3NzAyNjUxMWY0NzU2YmYxNGEyZWY3YWI2ODQyOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iZjAxZGZhZDc3MWQ0MDU0OTY1NjYxYTk4YjZhNDY1OCA9ICQoJzxkaXYgaWQ9Imh0bWxfYmYwMWRmYWQ3NzFkNDA1NDk2NTY2MWE5OGI2YTQ2NTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZpcnN0IENhbmFkaWFuIFBsYWNlLCBVbmRlcmdyb3VuZCBjaXR5LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zOWE3NzAyNjUxMWY0NzU2YmYxNGEyZWY3YWI2ODQyOS5zZXRDb250ZW50KGh0bWxfYmYwMWRmYWQ3NzFkNDA1NDk2NTY2MWE5OGI2YTQ2NTgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDA1ZjAxYjg0MTBlNDE0M2JhNGVkM2Q2NWNiNDE0MjcuYmluZFBvcHVwKHBvcHVwXzM5YTc3MDI2NTExZjQ3NTZiZjE0YTJlZjdhYjY4NDI5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzMwNDhkZThhZDIzYTRmMTRhNDA4MGJlNWI0OTJkMDlhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5NTQyLC03OS40MjI1NjM3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I1NWU5YzFkODg3YzRkMzZhYTcyY2M1ZmJhZWNjMTQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNmMTk0NDI2YTM0ODRhNjU4YmE4OWU2YjQ2NzAxODBiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA3ODY3ZmI5MjczNjQzMzNiYmQwZTc0Y2E2ZDVjYWE0ID0gJCgnPGRpdiBpZD0iaHRtbF8wNzg2N2ZiOTI3MzY0MzMzYmJkMGU3NGNhNmQ1Y2FhNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2hyaXN0aWUsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzNmMTk0NDI2YTM0ODRhNjU4YmE4OWU2YjQ2NzAxODBiLnNldENvbnRlbnQoaHRtbF8wNzg2N2ZiOTI3MzY0MzMzYmJkMGU3NGNhNmQ1Y2FhNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zMDQ4ZGU4YWQyM2E0ZjE0YTQwODBiZTViNDkyZDA5YS5iaW5kUG9wdXAocG9wdXBfM2YxOTQ0MjZhMzQ4NGE2NThiYTg5ZTZiNDY3MDE4MGIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmI1MDE2N2E1NTI3NDI5NWFmNDJmNTUyZGY5ZTAyMDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjkwMDUxMDAwMDAwMSwtNzkuNDQyMjU5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNTVlOWMxZDg4N2M0ZDM2YWE3MmNjNWZiYWVjYzE0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80MTQ5NWI3MzNjNmM0MGJkODM1M2YwOWQzOWFjYjg4ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iNzYyYzNiYzQyYzE0MTUwOTE0MjBlMDcxZTgwMTcxMCA9ICQoJzxkaXYgaWQ9Imh0bWxfYjc2MmMzYmM0MmMxNDE1MDkxNDIwZTA3MWU4MDE3MTAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvdmVyY291cnQgVmlsbGFnZSwgRHVmZmVyaW4sIFdlc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDE0OTViNzMzYzZjNDBiZDgzNTNmMDlkMzlhY2I4OGUuc2V0Q29udGVudChodG1sX2I3NjJjM2JjNDJjMTQxNTA5MTQyMGUwNzFlODAxNzEwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZiNTAxNjdhNTUyNzQyOTVhZjQyZjU1MmRmOWUwMjAxLmJpbmRQb3B1cChwb3B1cF80MTQ5NWI3MzNjNmM0MGJkODM1M2YwOWQzOWFjYjg4ZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84MTU1MDljOGRiY2U0YTdhOWVmYzdhYjIzZmIyZDA3ZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzkyNjcwMDAwMDAwNiwtNzkuNDE5NzQ5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNTVlOWMxZDg4N2M0ZDM2YWE3MmNjNWZiYWVjYzE0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zN2NiZTA4MGY1MjY0MDU0YTNkODJhYTM2YmRkNWFkNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81MTFkMmM3YjM4MDY0ODVjODFmM2U3OTFkMTA4Zjk2YSA9ICQoJzxkaXYgaWQ9Imh0bWxfNTExZDJjN2IzODA2NDg1YzgxZjNlNzkxZDEwOGY5NmEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRyaW5pdHksIExpdHRsZSBQb3J0dWdhbCwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zN2NiZTA4MGY1MjY0MDU0YTNkODJhYTM2YmRkNWFkNy5zZXRDb250ZW50KGh0bWxfNTExZDJjN2IzODA2NDg1YzgxZjNlNzkxZDEwOGY5NmEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODE1NTA5YzhkYmNlNGE3YTllZmM3YWIyM2ZiMmQwN2UuYmluZFBvcHVwKHBvcHVwXzM3Y2JlMDgwZjUyNjQwNTRhM2Q4MmFhMzZiZGQ1YWQ3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FiMDQwZWZjN2UyNTQxNWI5ODM3ZTMyNjEyYmQyZmNlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjM2ODQ3MiwtNzkuNDI4MTkxNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTk0Yzc4NmE2ODAzNDkyZTlmYTMzOTdjOTA0MDM5MjMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjgyYmUzMDgxNTUwNGU0ZmIzZTU4NjcxMDMxZTE1MDkgPSAkKCc8ZGl2IGlkPSJodG1sXzY4MmJlMzA4MTU1MDRlNGZiM2U1ODY3MTAzMWUxNTA5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYXJrZGFsZSBWaWxsYWdlLCBCcm9ja3RvbiwgRXhoaWJpdGlvbiBQbGFjZSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81OTRjNzg2YTY4MDM0OTJlOWZhMzM5N2M5MDQwMzkyMy5zZXRDb250ZW50KGh0bWxfNjgyYmUzMDgxNTUwNGU0ZmIzZTU4NjcxMDMxZTE1MDkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWIwNDBlZmM3ZTI1NDE1Yjk4MzdlMzI2MTJiZDJmY2UuYmluZFBvcHVwKHBvcHVwXzU5NGM3ODZhNjgwMzQ5MmU5ZmEzMzk3YzkwNDAzOTIzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJmZmZjMWE0NDAzYzQ4M2NiNmQ4NTM2MWFlMTcyNjk2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYxNjA4MywtNzkuNDY0NzYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjU1ZTljMWQ4ODdjNGQzNmFhNzJjYzVmYmFlY2MxNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzZkMmZkNDRmOGNhNGQyOGJhNDAzMDRjZjAxYTlmOGQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODVmMjkxMjQyNWMyNGMzYjk0OTE0NWJiNmU4YTcyM2UgPSAkKCc8ZGl2IGlkPSJodG1sXzg1ZjI5MTI0MjVjMjRjM2I5NDkxNDViYjZlOGE3MjNlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgSnVuY3Rpb24gU291dGgsIEhpZ2ggUGFyaywgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jNmQyZmQ0NGY4Y2E0ZDI4YmE0MDMwNGNmMDFhOWY4ZC5zZXRDb250ZW50KGh0bWxfODVmMjkxMjQyNWMyNGMzYjk0OTE0NWJiNmU4YTcyM2UpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmZmZmMxYTQ0MDNjNDgzY2I2ZDg1MzYxYWUxNzI2OTYuYmluZFBvcHVwKHBvcHVwX2M2ZDJmZDQ0ZjhjYTRkMjhiYTQwMzA0Y2YwMWE5ZjhkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBkNWNlNWI4NzkxYjQyNjc4MzE0N2QzODgwNDgzNzU2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4OTU5NywtNzkuNDU2MzI1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I1NWU5YzFkODg3YzRkMzZhYTcyY2M1ZmJhZWNjMTQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzlkYmUyYjBiNmMwODQ5MDNiZTI3OTA0ZmMyYTFjODZlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQyNmEwNWZlZTc5MjQzYzRiYjcyMmYzY2UwNGU2MTYxID0gJCgnPGRpdiBpZD0iaHRtbF80MjZhMDVmZWU3OTI0M2M0YmI3MjJmM2NlMDRlNjE2MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFya2RhbGUsIFJvbmNlc3ZhbGxlcywgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85ZGJlMmIwYjZjMDg0OTAzYmUyNzkwNGZjMmExYzg2ZS5zZXRDb250ZW50KGh0bWxfNDI2YTA1ZmVlNzkyNDNjNGJiNzIyZjNjZTA0ZTYxNjEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGQ1Y2U1Yjg3OTFiNDI2NzgzMTQ3ZDM4ODA0ODM3NTYuYmluZFBvcHVwKHBvcHVwXzlkYmUyYjBiNmMwODQ5MDNiZTI3OTA0ZmMyYTFjODZlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I3NzMxZDQyMjQwYjRmZjZiOTgzZmIzNWI5ZTExYzYwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNTcwNiwtNzkuNDg0NDQ5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNTVlOWMxZDg4N2M0ZDM2YWE3MmNjNWZiYWVjYzE0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81ZmIwMjNkNjJkMWU0MjUwYjRmZWZkNDk3YzhjNTc3OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mYjA4MTIxYjk0ZTc0Y2UxYWExY2MzNmJhMTZhZDY2ZiA9ICQoJzxkaXYgaWQ9Imh0bWxfZmIwODEyMWI5NGU3NGNlMWFhMWNjMzZiYTE2YWQ2NmYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN3YW5zZWEsIFJ1bm55bWVkZSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81ZmIwMjNkNjJkMWU0MjUwYjRmZWZkNDk3YzhjNTc3OC5zZXRDb250ZW50KGh0bWxfZmIwODEyMWI5NGU3NGNlMWFhMWNjMzZiYTE2YWQ2NmYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjc3MzFkNDIyNDBiNGZmNmI5ODNmYjM1YjllMTFjNjAuYmluZFBvcHVwKHBvcHVwXzVmYjAyM2Q2MmQxZTQyNTBiNGZlZmQ0OTdjOGM1Nzc4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI5NzA2NmJmMjBlMTRkZTFiZTRhYzFmN2NmMGYwMjVmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNzQzOSwtNzkuMzIxNTU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I1NWU5YzFkODg3YzRkMzZhYTcyY2M1ZmJhZWNjMTQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JhOGM5YzNjNjk4YjQxOGQ5ZWJjY2Y2MjIxMDA2YjdiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzM5YmI3MzM0M2Y4NTRkODY5Mjc0MjZiYWZlN2YwZTQ5ID0gJCgnPGRpdiBpZD0iaHRtbF8zOWJiNzMzNDNmODU0ZDg2OTI3NDI2YmFmZTdmMGU0OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QnVzaW5lc3MgUmVwbHkgTWFpbCBQcm9jZXNzaW5nIENlbnRyZSA5NjkgRWFzdGVybiwgRWFzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iYThjOWMzYzY5OGI0MThkOWViY2NmNjIyMTAwNmI3Yi5zZXRDb250ZW50KGh0bWxfMzliYjczMzQzZjg1NGQ4NjkyNzQyNmJhZmU3ZjBlNDkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjk3MDY2YmYyMGUxNGRlMWJlNGFjMWY3Y2YwZjAyNWYuYmluZFBvcHVwKHBvcHVwX2JhOGM5YzNjNjk4YjQxOGQ5ZWJjY2Y2MjIxMDA2YjdiKTsKCiAgICAgICAgICAgIAogICAgICAgIAo8L3NjcmlwdD4=" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



#### Define Foursquare Credentials and Version


```python
CLIENT_ID = 'R2KGTVRZKZ3VDRZALP1NALDNEC0OB2ORFFTYGSQNS42AVRE2' # your Foursquare ID
CLIENT_SECRET = 'FUVPPKJRWMGNAX3NXCKPVI1ULOPVRCPRMYOKFP0BNXWXVVAD' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
```

    Your credentails:
    CLIENT_ID: R2KGTVRZKZ3VDRZALP1NALDNEC0OB2ORFFTYGSQNS42AVRE2
    CLIENT_SECRET:FUVPPKJRWMGNAX3NXCKPVI1ULOPVRCPRMYOKFP0BNXWXVVAD


#### Let's explore the first neighborhood in our dataframe.

Get the neighborhood's name.


```python
toronto_df.loc[0, 'Neighbourhood']
```




    'The Beaches'



Get the neighborhood's latitude and longitude values.


```python
neighbourhood_latitude = toronto_df.loc[0, 'Latitude'] # neighbourhood latitude value
neighbourhood_longitude = toronto_df.loc[0, 'Longitude'] # neighbourhood longitude value

neighbourhood_name = toronto_df.loc[0, 'Neighbourhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighbourhood_name, 
                                                               neighbourhood_latitude, 
                                                               neighbourhood_longitude))
```

    Latitude and longitude values of The Beaches are 43.67635739999999, -79.2930312.


#### Now, let's get the top 100 venues that are in Marble Hill within a radius of 500 meters.

Let's create the GET request URL. 


```python
radius=500
LIMIT=100
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighbourhood_latitude, 
    neighbourhood_longitude, 
    radius, 
    LIMIT)
url
```




    'https://api.foursquare.com/v2/venues/explore?&client_id=R2KGTVRZKZ3VDRZALP1NALDNEC0OB2ORFFTYGSQNS42AVRE2&client_secret=FUVPPKJRWMGNAX3NXCKPVI1ULOPVRCPRMYOKFP0BNXWXVVAD&v=20180605&ll=43.67635739999999,-79.2930312&radius=500&limit=100'




```python
results = requests.get(url).json()
results
```




    {'meta': {'code': 200, 'requestId': '5ddad427f7706a001bf4f2c3'},
     'response': {'suggestedFilters': {'header': 'Tap to show:',
       'filters': [{'name': 'Open now', 'key': 'openNow'}]},
      'headerLocation': 'The Beaches',
      'headerFullLocation': 'The Beaches, Toronto',
      'headerLocationGranularity': 'neighborhood',
      'totalResults': 5,
      'suggestedBounds': {'ne': {'lat': 43.680857404499996,
        'lng': -79.28682091449052},
       'sw': {'lat': 43.67185739549999, 'lng': -79.29924148550948}},
      'groups': [{'type': 'Recommended Places',
        'name': 'recommended',
        'items': [{'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4bd461bc77b29c74a07d9282',
           'name': 'Glen Manor Ravine',
           'location': {'address': 'Glen Manor',
            'crossStreet': 'Queen St.',
            'lat': 43.67682094413784,
            'lng': -79.29394208780985,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.67682094413784,
              'lng': -79.29394208780985}],
            'distance': 89,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['Glen Manor (Queen St.)',
             'Toronto ON',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d159941735',
             'name': 'Trail',
             'pluralName': 'Trails',
             'shortName': 'Trail',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/parks_outdoors/hikingtrail_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4bd461bc77b29c74a07d9282-0'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad4c062f964a52011f820e3',
           'name': 'The Big Carrot Natural Food Market',
           'location': {'address': '125 Southwood Dr',
            'lat': 43.678879,
            'lng': -79.297734,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.678879,
              'lng': -79.297734}],
            'distance': 471,
            'postalCode': 'M4E 0B8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['125 Southwood Dr',
             'Toronto ON M4E 0B8',
             'Canada']},
           'categories': [{'id': '50aa9e744b90af0d42d5de0e',
             'name': 'Health Food Store',
             'pluralName': 'Health Food Stores',
             'shortName': 'Health Food Store',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/food_grocery_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '75150878'}},
          'referralId': 'e-0-4ad4c062f964a52011f820e3-1'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4b8daea1f964a520480833e3',
           'name': 'Grover Pub and Grub',
           'location': {'address': '676 Kingston Rd.',
            'crossStreet': 'at Main St.',
            'lat': 43.679181434941015,
            'lng': -79.29721535878515,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.679181434941015,
              'lng': -79.29721535878515}],
            'distance': 460,
            'postalCode': 'M4E 1R4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['676 Kingston Rd. (at Main St.)',
             'Toronto ON M4E 1R4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d11b941735',
             'name': 'Pub',
             'pluralName': 'Pubs',
             'shortName': 'Pub',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/nightlife/pub_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4b8daea1f964a520480833e3-2'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '56afcad6498e05333bf42031',
           'name': 'Glen Stewart Ravine',
           'location': {'lat': 43.67629984029563,
            'lng': -79.2947841389563,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.67629984029563,
              'lng': -79.2947841389563}],
            'distance': 141,
            'cc': 'CA',
            'country': 'Canada',
            'formattedAddress': ['Canada']},
           'categories': [{'id': '4bf58dd8d48988d162941735',
             'name': 'Other Great Outdoors',
             'pluralName': 'Other Great Outdoors',
             'shortName': 'Other Outdoors',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/parks_outdoors/outdoors_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-56afcad6498e05333bf42031-3'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4df91c4bae60f95f82229ad5',
           'name': 'Upper Beaches',
           'location': {'lat': 43.68056321147582,
            'lng': -79.2928688743688,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.68056321147582,
              'lng': -79.2928688743688}],
            'distance': 468,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['Toronto ON', 'Canada']},
           'categories': [{'id': '4f2a25ac4b909258e854f55f',
             'name': 'Neighborhood',
             'pluralName': 'Neighborhoods',
             'shortName': 'Neighborhood',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/parks_outdoors/neighborhood_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4df91c4bae60f95f82229ad5-4'}]}]}}




```python
# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
```


```python
venues = results['response']['groups'][0]['items']
```

Now we are ready to clean the json and structure it into a *pandas* dataframe.


```python
#venues = results['response']['groups'][0]['items']
  
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>categories</th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Glen Manor Ravine</td>
      <td>Trail</td>
      <td>43.676821</td>
      <td>-79.293942</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Big Carrot Natural Food Market</td>
      <td>Health Food Store</td>
      <td>43.678879</td>
      <td>-79.297734</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grover Pub and Grub</td>
      <td>Pub</td>
      <td>43.679181</td>
      <td>-79.297215</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Glen Stewart Ravine</td>
      <td>Other Great Outdoors</td>
      <td>43.676300</td>
      <td>-79.294784</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Upper Beaches</td>
      <td>Neighborhood</td>
      <td>43.680563</td>
      <td>-79.292869</td>
    </tr>
  </tbody>
</table>
</div>



And how many venues were returned by Foursquare?


```python
print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))
```

    5 venues were returned by Foursquare.


#### Let's create a function to repeat the same process to all the neighborhoods in Toronto


```python
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood', 
                  'Neighbourhood Latitude', 
                  'Neighbourhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
```

#### Run the above function on each neighborhood and create a new dataframe called *toronto_venues*.


```python
toronto_venues = getNearbyVenues(names=toronto_df['Neighbourhood'],
                                   latitudes=toronto_df['Latitude'],
                                   longitudes=toronto_df['Longitude']
                                  )
```

    The Beaches
    The Danforth West, Riverdale
    India Bazaar, The Beaches West
    Studio District
    Lawrence Park
    Davisville North
    North Toronto West
    Davisville
    Moore Park, Summerhill East
    Deer Park, Forest Hill SE, Rathnelly, Summerhill West, South Hill
    Rosedale
    Cabbagetown, St. James Town
    Church and Wellesley
    Harbourfront
    Ryerson, Garden District
    St. James Town
    Berczy Park
    Central Bay Street
    King, Richmond, Adelaide
    Toronto Islands, Union Station, Harbourfront East
    Toronto Dominion Centre, Design Exchange
    Victoria Hotel, Commerce Court
    Roselawn
    Forest Hill West, Forest Hill North
    North Midtown, The Annex, Yorkville
    Harbord, University of Toronto
    Grange Park, Chinatown, Kensington Market
    Bathurst Quay, Harbourfront West, Island airport, CN Tower, South Niagara, Railway Lands, King and Spadina
    Stn A PO Boxes 25 The Esplanade
    First Canadian Place, Underground city
    Christie
    Dovercourt Village, Dufferin
    Trinity, Little Portugal
    Parkdale Village, Brockton, Exhibition Place
    The Junction South, High Park
    Parkdale, Roncesvalles
    Swansea, Runnymede
    Business Reply Mail Processing Centre 969 Eastern


#### Let's check the size of the resulting dataframe


```python
print(toronto_venues.shape)
toronto_venues.head()
```

    (1693, 7)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Neighbourhood Latitude</th>
      <th>Neighbourhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Glen Manor Ravine</td>
      <td>43.676821</td>
      <td>-79.293942</td>
      <td>Trail</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>The Big Carrot Natural Food Market</td>
      <td>43.678879</td>
      <td>-79.297734</td>
      <td>Health Food Store</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Grover Pub and Grub</td>
      <td>43.679181</td>
      <td>-79.297215</td>
      <td>Pub</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Glen Stewart Ravine</td>
      <td>43.676300</td>
      <td>-79.294784</td>
      <td>Other Great Outdoors</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Upper Beaches</td>
      <td>43.680563</td>
      <td>-79.292869</td>
      <td>Neighborhood</td>
    </tr>
  </tbody>
</table>
</div>



Let's check how many venues were returned for each neighborhood


```python
toronto_venues.groupby('Neighbourhood').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood Latitude</th>
      <th>Neighbourhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
    <tr>
      <th>Neighbourhood</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bathurst Quay, Harbourfront West, Island airport, CN Tower, South Niagara, Railway Lands, King and Spadina</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>Berczy Park</th>
      <td>57</td>
      <td>57</td>
      <td>57</td>
      <td>57</td>
      <td>57</td>
      <td>57</td>
    </tr>
    <tr>
      <th>Business Reply Mail Processing Centre 969 Eastern</th>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
    </tr>
    <tr>
      <th>Cabbagetown, St. James Town</th>
      <td>45</td>
      <td>45</td>
      <td>45</td>
      <td>45</td>
      <td>45</td>
      <td>45</td>
    </tr>
    <tr>
      <th>Central Bay Street</th>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
    </tr>
    <tr>
      <th>Christie</th>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
    </tr>
    <tr>
      <th>Church and Wellesley</th>
      <td>85</td>
      <td>85</td>
      <td>85</td>
      <td>85</td>
      <td>85</td>
      <td>85</td>
    </tr>
    <tr>
      <th>Davisville</th>
      <td>38</td>
      <td>38</td>
      <td>38</td>
      <td>38</td>
      <td>38</td>
      <td>38</td>
    </tr>
    <tr>
      <th>Davisville North</th>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Deer Park, Forest Hill SE, Rathnelly, Summerhill West, South Hill</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>Dovercourt Village, Dufferin</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>First Canadian Place, Underground city</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Forest Hill West, Forest Hill North</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Grange Park, Chinatown, Kensington Market</th>
      <td>95</td>
      <td>95</td>
      <td>95</td>
      <td>95</td>
      <td>95</td>
      <td>95</td>
    </tr>
    <tr>
      <th>Harbord, University of Toronto</th>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Harbourfront</th>
      <td>51</td>
      <td>51</td>
      <td>51</td>
      <td>51</td>
      <td>51</td>
      <td>51</td>
    </tr>
    <tr>
      <th>India Bazaar, The Beaches West</th>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
    </tr>
    <tr>
      <th>King, Richmond, Adelaide</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Lawrence Park</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Moore Park, Summerhill East</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>North Midtown, The Annex, Yorkville</th>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
    </tr>
    <tr>
      <th>North Toronto West</th>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
    </tr>
    <tr>
      <th>Parkdale Village, Brockton, Exhibition Place</th>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
    </tr>
    <tr>
      <th>Parkdale, Roncesvalles</th>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>Rosedale</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Roselawn</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Ryerson, Garden District</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>St. James Town</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Stn A PO Boxes 25 The Esplanade</th>
      <td>98</td>
      <td>98</td>
      <td>98</td>
      <td>98</td>
      <td>98</td>
      <td>98</td>
    </tr>
    <tr>
      <th>Studio District</th>
      <td>38</td>
      <td>38</td>
      <td>38</td>
      <td>38</td>
      <td>38</td>
      <td>38</td>
    </tr>
    <tr>
      <th>Swansea, Runnymede</th>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
    </tr>
    <tr>
      <th>The Beaches</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>The Danforth West, Riverdale</th>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
    </tr>
    <tr>
      <th>The Junction South, High Park</th>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
    </tr>
    <tr>
      <th>Toronto Dominion Centre, Design Exchange</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Toronto Islands, Union Station, Harbourfront East</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Trinity, Little Portugal</th>
      <td>64</td>
      <td>64</td>
      <td>64</td>
      <td>64</td>
      <td>64</td>
      <td>64</td>
    </tr>
    <tr>
      <th>Victoria Hotel, Commerce Court</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



#### Let's find out how many unique categories can be curated from all the returned venues


```python
print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))
```

    There are 232 uniques categories.


<a id='item3'></a>

## 3. Analyze Each Neighborhood


```python
# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighbourhood column back to dataframe
toronto_onehot['Neighbourhood'] = toronto_venues['Neighbourhood'] 

# move neighbourhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>...</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Wings Joint</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 233 columns</p>
</div>



And let's examine the new dataframe size.


```python
toronto_onehot.shape
```




    (1693, 233)



#### Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category


```python
toronto_grouped = toronto_onehot.groupby('Neighbourhood').mean().reset_index()
toronto_grouped
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>...</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Wings Joint</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bathurst Quay, Harbourfront West, Island airpo...</td>
      <td>0.000000</td>
      <td>0.0625</td>
      <td>0.0625</td>
      <td>0.0625</td>
      <td>0.125</td>
      <td>0.1875</td>
      <td>0.0625</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Berczy Park</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.017544</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cabbagetown, St. James Town</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Central Bay Street</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Christie</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Church and Wellesley</td>
      <td>0.011765</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.011765</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011765</td>
      <td>0.000000</td>
      <td>0.011765</td>
      <td>0.011765</td>
      <td>0.011765</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Davisville</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.026316</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Davisville North</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Deer Park, Forest Hill SE, Rathnelly, Summerhi...</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Dovercourt Village, Dufferin</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>First Canadian Place, Underground city</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Forest Hill West, Forest Hill North</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.25000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Grange Park, Chinatown, Kensington Market</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.031579</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.010526</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Harbord, University of Toronto</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Harbourfront</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.019608</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.019608</td>
    </tr>
    <tr>
      <th>16</th>
      <td>India Bazaar, The Beaches West</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>King, Richmond, Adelaide</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Lawrence Park</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Moore Park, Summerhill East</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>North Midtown, The Annex, Yorkville</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>North Toronto West</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047619</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Parkdale Village, Brockton, Exhibition Place</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Parkdale, Roncesvalles</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Rosedale</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.25000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Roselawn</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Ryerson, Garden District</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.010000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>St. James Town</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Stn A PO Boxes 25 The Esplanade</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.010204</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.010204</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Studio District</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Swansea, Runnymede</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>The Beaches</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.20000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>The Danforth West, Riverdale</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.02381</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023810</td>
    </tr>
    <tr>
      <th>33</th>
      <td>The Junction South, High Park</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Toronto Dominion Centre, Design Exchange</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Toronto Islands, Union Station, Harbourfront East</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Trinity, Little Portugal</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.015625</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.015625</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015625</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Victoria Hotel, Commerce Court</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>38 rows × 233 columns</p>
</div>



#### Let's confirm the new size


```python
toronto_grouped.shape
```




    (38, 233)



#### Let's print each neighborhood along with the top 5 most common venues


```python
num_top_venues = 5

for hood in toronto_grouped['Neighbourhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighbourhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')
```

    ----Bathurst Quay, Harbourfront West, Island airport, CN Tower, South Niagara, Railway Lands, King and Spadina----
                  venue  freq
    0   Airport Service  0.19
    1    Airport Lounge  0.12
    2  Sculpture Garden  0.06
    3       Coffee Shop  0.06
    4          Boutique  0.06
    
    
    ----Berczy Park----
                    venue  freq
    0         Coffee Shop  0.09
    1              Bakery  0.05
    2      Farmers Market  0.04
    3  Seafood Restaurant  0.04
    4            Beer Bar  0.04
    
    
    ----Business Reply Mail Processing Centre 969 Eastern----
                    venue  freq
    0  Light Rail Station  0.11
    1         Yoga Studio  0.05
    2                Park  0.05
    3          Comic Shop  0.05
    4    Recording Studio  0.05
    
    
    ----Cabbagetown, St. James Town----
             venue  freq
    0  Coffee Shop  0.07
    1       Bakery  0.07
    2   Restaurant  0.04
    3          Pub  0.04
    4         Café  0.04
    
    
    ----Central Bay Street----
                    venue  freq
    0         Coffee Shop  0.15
    1  Italian Restaurant  0.05
    2      Ice Cream Shop  0.05
    3        Burger Joint  0.04
    4      Sandwich Place  0.04
    
    
    ----Christie----
                   venue  freq
    0      Grocery Store  0.18
    1               Café  0.18
    2               Park  0.12
    3  Convenience Store  0.06
    4        Candy Store  0.06
    
    
    ----Church and Wellesley----
                     venue  freq
    0          Coffee Shop  0.08
    1     Sushi Restaurant  0.06
    2  Japanese Restaurant  0.06
    3           Restaurant  0.04
    4              Gay Bar  0.04
    
    
    ----Davisville----
                venue  freq
    0     Pizza Place  0.08
    1    Dessert Shop  0.08
    2  Sandwich Place  0.08
    3     Coffee Shop  0.05
    4             Gym  0.05
    
    
    ----Davisville North----
                   venue  freq
    0     Breakfast Spot  0.14
    1  Food & Drink Shop  0.14
    2     Clothing Store  0.14
    3                Gym  0.14
    4               Park  0.14
    
    
    ----Deer Park, Forest Hill SE, Rathnelly, Summerhill West, South Hill----
                     venue  freq
    0          Coffee Shop  0.12
    1                  Pub  0.12
    2           Sports Bar  0.06
    3  Fried Chicken Joint  0.06
    4     Sushi Restaurant  0.06
    
    
    ----Dovercourt Village, Dufferin----
             venue  freq
    0       Bakery  0.14
    1     Pharmacy  0.14
    2  Supermarket  0.14
    3         Bank  0.07
    4          Bar  0.07
    
    
    ----First Canadian Place, Underground city----
              venue  freq
    0   Coffee Shop  0.11
    1          Café  0.07
    2    Steakhouse  0.04
    3         Hotel  0.04
    4  Burger Joint  0.03
    
    
    ----Forest Hill West, Forest Hill North----
                    venue  freq
    0  Mexican Restaurant  0.25
    1               Trail  0.25
    2    Sushi Restaurant  0.25
    3       Jewelry Store  0.25
    4   Afghan Restaurant  0.00
    
    
    ----Grange Park, Chinatown, Kensington Market----
                       venue  freq
    0                   Café  0.08
    1                    Bar  0.06
    2  Vietnamese Restaurant  0.05
    3            Coffee Shop  0.04
    4     Mexican Restaurant  0.04
    
    
    ----Harbord, University of Toronto----
                venue  freq
    0            Café  0.14
    1             Bar  0.06
    2  Sandwich Place  0.06
    3      Restaurant  0.06
    4          Bakery  0.06
    
    
    ----Harbourfront----
             venue  freq
    0  Coffee Shop  0.16
    1         Park  0.08
    2          Pub  0.06
    3       Bakery  0.06
    4      Theater  0.04
    
    
    ----India Bazaar, The Beaches West----
                venue  freq
    0            Park  0.12
    1         Brewery  0.06
    2  Sandwich Place  0.06
    3    Liquor Store  0.06
    4      Steakhouse  0.06
    
    
    ----King, Richmond, Adelaide----
                 venue  freq
    0      Coffee Shop  0.07
    1             Café  0.05
    2  Thai Restaurant  0.04
    3       Steakhouse  0.04
    4              Bar  0.04
    
    
    ----Lawrence Park----
                         venue  freq
    0                     Park  0.33
    1              Swim School  0.33
    2                 Bus Line  0.33
    3        Afghan Restaurant  0.00
    4  New American Restaurant  0.00
    
    
    ----Moore Park, Summerhill East----
                         venue  freq
    0             Tennis Court  0.33
    1                      Gym  0.33
    2               Playground  0.33
    3                 Pharmacy  0.00
    4  New American Restaurant  0.00
    
    
    ----North Midtown, The Annex, Yorkville----
                venue  freq
    0            Café  0.14
    1  Sandwich Place  0.14
    2     Coffee Shop  0.10
    3      Donut Shop  0.05
    4       BBQ Joint  0.05
    
    
    ----North Toronto West----
                        venue  freq
    0             Coffee Shop  0.10
    1          Clothing Store  0.10
    2     Sporting Goods Shop  0.10
    3             Yoga Studio  0.05
    4  Furniture / Home Store  0.05
    
    
    ----Parkdale Village, Brockton, Exhibition Place----
                venue  freq
    0  Breakfast Spot  0.10
    1            Café  0.10
    2     Coffee Shop  0.10
    3   Grocery Store  0.05
    4         Stadium  0.05
    
    
    ----Parkdale, Roncesvalles----
                             venue  freq
    0                    Gift Shop  0.13
    1                  Coffee Shop  0.13
    2  Eastern European Restaurant  0.07
    3                Movie Theater  0.07
    4                         Bank  0.07
    
    
    ----Rosedale----
                          venue  freq
    0                      Park  0.50
    1                Playground  0.25
    2                     Trail  0.25
    3   New American Restaurant  0.00
    4  Mediterranean Restaurant  0.00
    
    
    ----Roselawn----
                           venue  freq
    0                     Garden   1.0
    1          Afghan Restaurant   0.0
    2                  Nightclub   0.0
    3         Mexican Restaurant   0.0
    4  Middle Eastern Restaurant   0.0
    
    
    ----Ryerson, Garden District----
                      venue  freq
    0           Coffee Shop  0.08
    1        Clothing Store  0.07
    2                  Café  0.03
    3  Fast Food Restaurant  0.03
    4        Cosmetics Shop  0.03
    
    
    ----St. James Town----
                venue  freq
    0            Café  0.06
    1     Coffee Shop  0.06
    2           Hotel  0.05
    3      Restaurant  0.05
    4  Clothing Store  0.04
    
    
    ----Stn A PO Boxes 25 The Esplanade----
                     venue  freq
    0          Coffee Shop  0.10
    1           Restaurant  0.04
    2                 Café  0.04
    3   Seafood Restaurant  0.03
    4  Japanese Restaurant  0.03
    
    
    ----Studio District----
                     venue  freq
    0                 Café  0.11
    1          Coffee Shop  0.08
    2          Yoga Studio  0.05
    3   Italian Restaurant  0.05
    4  American Restaurant  0.05
    
    
    ----Swansea, Runnymede----
                    venue  freq
    0         Coffee Shop  0.09
    1                Café  0.09
    2  Italian Restaurant  0.06
    3    Sushi Restaurant  0.06
    4      Ice Cream Shop  0.03
    
    
    ----The Beaches----
                      venue  freq
    0     Health Food Store   0.2
    1                   Pub   0.2
    2                 Trail   0.2
    3  Other Great Outdoors   0.2
    4          Neighborhood   0.2
    
    
    ----The Danforth West, Riverdale----
                        venue  freq
    0        Greek Restaurant  0.21
    1             Coffee Shop  0.10
    2          Ice Cream Shop  0.07
    3      Italian Restaurant  0.07
    4  Furniture / Home Store  0.05
    
    
    ----The Junction South, High Park----
                     venue  freq
    0                 Café  0.08
    1   Mexican Restaurant  0.08
    2                  Bar  0.08
    3      Thai Restaurant  0.08
    4  Arts & Crafts Store  0.04
    
    
    ----Toronto Dominion Centre, Design Exchange----
             venue  freq
    0  Coffee Shop  0.13
    1         Café  0.08
    2        Hotel  0.07
    3   Restaurant  0.04
    4          Bar  0.04
    
    
    ----Toronto Islands, Union Station, Harbourfront East----
                venue  freq
    0     Coffee Shop  0.13
    1           Hotel  0.05
    2        Aquarium  0.05
    3            Café  0.04
    4  Scenic Lookout  0.03
    
    
    ----Trinity, Little Portugal----
                  venue  freq
    0               Bar  0.09
    1       Coffee Shop  0.06
    2  Asian Restaurant  0.05
    3        Restaurant  0.05
    4       Men's Store  0.05
    
    
    ----Victoria Hotel, Commerce Court----
             venue  freq
    0  Coffee Shop  0.11
    1         Café  0.07
    2        Hotel  0.06
    3   Restaurant  0.04
    4          Gym  0.03
    
    


#### Let's put that into a *pandas* dataframe

First, let's write a function to sort the venues in descending order.


```python
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
```

Now let's create the new dataframe and display the top 10 venues for each neighborhood.


```python
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighbourhoods_venues_sorted = pd.DataFrame(columns=columns)
neighbourhoods_venues_sorted['Neighbourhood'] = toronto_grouped['Neighbourhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighbourhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighbourhoods_venues_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bathurst Quay, Harbourfront West, Island airpo...</td>
      <td>Airport Service</td>
      <td>Airport Lounge</td>
      <td>Harbor / Marina</td>
      <td>Bar</td>
      <td>Plane</td>
      <td>Coffee Shop</td>
      <td>Sculpture Garden</td>
      <td>Boat or Ferry</td>
      <td>Boutique</td>
      <td>Airport Gate</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Berczy Park</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Cocktail Bar</td>
      <td>Beer Bar</td>
      <td>Farmers Market</td>
      <td>Steakhouse</td>
      <td>Café</td>
      <td>Cheese Shop</td>
      <td>Seafood Restaurant</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>Light Rail Station</td>
      <td>Yoga Studio</td>
      <td>Spa</td>
      <td>Garden Center</td>
      <td>Garden</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Comic Shop</td>
      <td>Park</td>
      <td>Recording Studio</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cabbagetown, St. James Town</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Italian Restaurant</td>
      <td>Café</td>
      <td>Pizza Place</td>
      <td>Pub</td>
      <td>Chinese Restaurant</td>
      <td>Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Snack Place</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Central Bay Street</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Ice Cream Shop</td>
      <td>Sandwich Place</td>
      <td>Café</td>
      <td>Burger Joint</td>
      <td>Bakery</td>
      <td>Bar</td>
      <td>Chinese Restaurant</td>
      <td>Spa</td>
    </tr>
  </tbody>
</table>
</div>



<a id='item4'></a>

## 4. Cluster Neighborhoods

Run *k*-means to cluster the neighborhood into 5 clusters.


```python
# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)



Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.


```python
# add clustering labels
neighbourhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = toronto_df

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighbourhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

toronto_merged.head() # check the last columns!
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postcode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>0</td>
      <td>Health Food Store</td>
      <td>Other Great Outdoors</td>
      <td>Pub</td>
      <td>Trail</td>
      <td>Neighborhood</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
      <td>Event Space</td>
      <td>Falafel Restaurant</td>
      <td>Dim Sum Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>0</td>
      <td>Greek Restaurant</td>
      <td>Coffee Shop</td>
      <td>Ice Cream Shop</td>
      <td>Italian Restaurant</td>
      <td>Furniture / Home Store</td>
      <td>Restaurant</td>
      <td>Grocery Store</td>
      <td>Brewery</td>
      <td>Bubble Tea Shop</td>
      <td>Caribbean Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>India Bazaar, The Beaches West</td>
      <td>43.668999</td>
      <td>-79.315572</td>
      <td>0</td>
      <td>Park</td>
      <td>Gym</td>
      <td>Italian Restaurant</td>
      <td>Pet Store</td>
      <td>Pub</td>
      <td>Movie Theater</td>
      <td>Sandwich Place</td>
      <td>Burrito Place</td>
      <td>Burger Joint</td>
      <td>Brewery</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>43.659526</td>
      <td>-79.340923</td>
      <td>0</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Yoga Studio</td>
      <td>Italian Restaurant</td>
      <td>American Restaurant</td>
      <td>Bakery</td>
      <td>Neighborhood</td>
      <td>Diner</td>
      <td>Comfort Food Restaurant</td>
      <td>Clothing Store</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>43.728020</td>
      <td>-79.388790</td>
      <td>2</td>
      <td>Park</td>
      <td>Swim School</td>
      <td>Bus Line</td>
      <td>Yoga Studio</td>
      <td>Dog Run</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



Finally, let's visualize the resulting clusters


```python
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighbourhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfZTcwOWZkNDUxNmE2NDJjY2I3NjliOWVhZGQ4NDg5MzkgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2U3MDlmZDQ1MTZhNjQyY2NiNzY5YjllYWRkODQ4OTM5IiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOScsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUzOTYzLC03OS4zODcyMDddLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgem9vbTogMTEsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYXhCb3VuZHM6IGJvdW5kcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxheWVyczogW10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3b3JsZENvcHlKdW1wOiBmYWxzZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSk7CiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyX2M5ODRjMTU1MjRjMjRmYjg5YTBiMzk3ZWQxNmQxMzg2ID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgICAgICAgICAgewogICJhdHRyaWJ1dGlvbiI6IG51bGwsCiAgImRldGVjdFJldGluYSI6IGZhbHNlLAogICJtYXhab29tIjogMTgsCiAgIm1pblpvb20iOiAxLAogICJub1dyYXAiOiBmYWxzZSwKICAic3ViZG9tYWlucyI6ICJhYmMiCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U3MDlmZDQ1MTZhNjQyY2NiNzY5YjllYWRkODQ4OTM5KTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82YTU4OGViZWE2YzI0YzA1YjM3YzI5ZWMyNWY2Mzc2NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3NjM1NzM5OTk5OTk5LC03OS4yOTMwMzEyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U3MDlmZDQ1MTZhNjQyY2NiNzY5YjllYWRkODQ4OTM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E4MDZlYTFiZGY4MzQzNmU5ZjM3ZWM4N2M5NDhkZmRmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2NlNjg5NjE1ZjZkNDRkYzg5NmU0YzhkYzkzMTdmODk5ID0gJCgnPGRpdiBpZD0iaHRtbF9jZTY4OTYxNWY2ZDQ0ZGM4OTZlNGM4ZGM5MzE3Zjg5OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEJlYWNoZXMgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hODA2ZWExYmRmODM0MzZlOWYzN2VjODdjOTQ4ZGZkZi5zZXRDb250ZW50KGh0bWxfY2U2ODk2MTVmNmQ0NGRjODk2ZTRjOGRjOTMxN2Y4OTkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNmE1ODhlYmVhNmMyNGMwNWIzN2MyOWVjMjVmNjM3NjYuYmluZFBvcHVwKHBvcHVwX2E4MDZlYTFiZGY4MzQzNmU5ZjM3ZWM4N2M5NDhkZmRmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ0YWFlY2U0N2UxZTRmMTRhZjEzMDViNzFmMjc2ZDNiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc5NTU3MSwtNzkuMzUyMTg4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U3MDlmZDQ1MTZhNjQyY2NiNzY5YjllYWRkODQ4OTM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2M3NTVhZTFmYTU3MTQ5YmZhZTA5NGI5YWYxNDZiZTM4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzIzOGM4OWRkY2U3YzRkY2ZhMGI1YTQ3MGIyOTI1OGNlID0gJCgnPGRpdiBpZD0iaHRtbF8yMzhjODlkZGNlN2M0ZGNmYTBiNWE0NzBiMjkyNThjZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIERhbmZvcnRoIFdlc3QsIFJpdmVyZGFsZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M3NTVhZTFmYTU3MTQ5YmZhZTA5NGI5YWYxNDZiZTM4LnNldENvbnRlbnQoaHRtbF8yMzhjODlkZGNlN2M0ZGNmYTBiNWE0NzBiMjkyNThjZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80NGFhZWNlNDdlMWU0ZjE0YWYxMzA1YjcxZjI3NmQzYi5iaW5kUG9wdXAocG9wdXBfYzc1NWFlMWZhNTcxNDliZmFlMDk0YjlhZjE0NmJlMzgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYmIwMTEwYzJhNzlkNGY0YWEyNjVjMjJmZTNlYTdjYTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njg5OTg1LC03OS4zMTU1NzE1OTk5OTk5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kMTgxNzY2NzIwMWU0OTg1YTJjYTE0OWVkNjNhYzcyMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84MWYxMDQyNjZiZjA0OTYyYmI5MDc2NTUyOThkNThmZCA9ICQoJzxkaXYgaWQ9Imh0bWxfODFmMTA0MjY2YmYwNDk2MmJiOTA3NjU1Mjk4ZDU4ZmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkluZGlhIEJhemFhciwgVGhlIEJlYWNoZXMgV2VzdCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QxODE3NjY3MjAxZTQ5ODVhMmNhMTQ5ZWQ2M2FjNzIwLnNldENvbnRlbnQoaHRtbF84MWYxMDQyNjZiZjA0OTYyYmI5MDc2NTUyOThkNThmZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iYjAxMTBjMmE3OWQ0ZjRhYTI2NWMyMmZlM2VhN2NhMC5iaW5kUG9wdXAocG9wdXBfZDE4MTc2NjcyMDFlNDk4NWEyY2ExNDllZDYzYWM3MjApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTMyNjUxMWM3NTc2NGZkMThkZGI3MzdhNDM4ZDczZGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTk1MjU1LC03OS4zNDA5MjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTcwOWZkNDUxNmE2NDJjY2I3NjliOWVhZGQ4NDg5MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTk5MmU4ZWZiMDZlNDdmMGE2N2Y2MDIyOTM5YTcyYWQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTNkZmIyNGRlMjJkNGQ2NDg1MmY1MGQ3NDVkNmE0MTggPSAkKCc8ZGl2IGlkPSJodG1sX2UzZGZiMjRkZTIyZDRkNjQ4NTJmNTBkNzQ1ZDZhNDE4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdHVkaW8gRGlzdHJpY3QgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hOTkyZThlZmIwNmU0N2YwYTY3ZjYwMjI5MzlhNzJhZC5zZXRDb250ZW50KGh0bWxfZTNkZmIyNGRlMjJkNGQ2NDg1MmY1MGQ3NDVkNmE0MTgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTMyNjUxMWM3NTc2NGZkMThkZGI3MzdhNDM4ZDczZGUuYmluZFBvcHVwKHBvcHVwX2E5OTJlOGVmYjA2ZTQ3ZjBhNjdmNjAyMjkzOWE3MmFkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdkMjQ1YTU5MTQ1OTQ5MTRhZTA0ZDYyMjk5NjgyMDIzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI4MDIwNSwtNzkuMzg4NzkwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMDBiNWViIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzAwYjVlYiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81NzU5YmFlOTE4NDI0YWM2YjJhMGU2NmJjZjk1ODY2OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83NjhlOTQzZWMzZGY0OTFkYjY2NmU4NGJiOTY5YjU5ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfNzY4ZTk0M2VjM2RmNDkxZGI2NjZlODRiYjk2OWI1OWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxhd3JlbmNlIFBhcmsgQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81NzU5YmFlOTE4NDI0YWM2YjJhMGU2NmJjZjk1ODY2OC5zZXRDb250ZW50KGh0bWxfNzY4ZTk0M2VjM2RmNDkxZGI2NjZlODRiYjk2OWI1OWQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2QyNDVhNTkxNDU5NDkxNGFlMDRkNjIyOTk2ODIwMjMuYmluZFBvcHVwKHBvcHVwXzU3NTliYWU5MTg0MjRhYzZiMmEwZTY2YmNmOTU4NjY4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VmMjQzYWZmODBmYTQ5YjBhNDE1YWMyZWMxODBjMzY4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzEyNzUxMSwtNzkuMzkwMTk3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hZmI3ZjI5ZjAwZmM0NDU0YmVlMTRjN2U4Y2JlZWJjYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82Y2Y4NTM3MmUxMTk0OGIxOWUzN2I3YzI1YjM5NWZhZiA9ICQoJzxkaXYgaWQ9Imh0bWxfNmNmODUzNzJlMTE5NDhiMTllMzdiN2MyNWIzOTVmYWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRhdmlzdmlsbGUgTm9ydGggQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hZmI3ZjI5ZjAwZmM0NDU0YmVlMTRjN2U4Y2JlZWJjYS5zZXRDb250ZW50KGh0bWxfNmNmODUzNzJlMTE5NDhiMTllMzdiN2MyNWIzOTVmYWYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZWYyNDNhZmY4MGZhNDliMGE0MTVhYzJlYzE4MGMzNjguYmluZFBvcHVwKHBvcHVwX2FmYjdmMjlmMDBmYzQ0NTRiZWUxNGM3ZThjYmVlYmNhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I4ZjNkNTQ2MDk2NzRjNTc4ZTE3MzljZGFkYzg3ODBhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzE1MzgzNCwtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTcwOWZkNDUxNmE2NDJjY2I3NjliOWVhZGQ4NDg5MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDY1MjQxNTY5ZDE2NDhjN2IxY2NmNGE4N2UzYzgyMzkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmY0NzAzY2QxYTM5NGMxZjk5Y2E4OTkzZTQ3NmFhY2MgPSAkKCc8ZGl2IGlkPSJodG1sX2JmNDcwM2NkMWEzOTRjMWY5OWNhODk5M2U0NzZhYWNjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aCBUb3JvbnRvIFdlc3QgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wNjUyNDE1NjlkMTY0OGM3YjFjY2Y0YTg3ZTNjODIzOS5zZXRDb250ZW50KGh0bWxfYmY0NzAzY2QxYTM5NGMxZjk5Y2E4OTkzZTQ3NmFhY2MpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjhmM2Q1NDYwOTY3NGM1NzhlMTczOWNkYWRjODc4MGEuYmluZFBvcHVwKHBvcHVwXzA2NTI0MTU2OWQxNjQ4YzdiMWNjZjRhODdlM2M4MjM5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZiNDVlZjlkN2Q0ZjRiNTk5ZjFmNzZhNGEzMmU0ZTViID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA0MzI0NCwtNzkuMzg4NzkwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mNTExYTAxMWI2YTQ0ZGRjOGZmMmIyOGZlMTZiYjM3ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mMTRiZmQwNzhiZDI0ZWViYTNmZmRjNDI4NmM1YjE5NCA9ICQoJzxkaXYgaWQ9Imh0bWxfZjE0YmZkMDc4YmQyNGVlYmEzZmZkYzQyODZjNWIxOTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRhdmlzdmlsbGUgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mNTExYTAxMWI2YTQ0ZGRjOGZmMmIyOGZlMTZiYjM3ZS5zZXRDb250ZW50KGh0bWxfZjE0YmZkMDc4YmQyNGVlYmEzZmZkYzQyODZjNWIxOTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmI0NWVmOWQ3ZDRmNGI1OTlmMWY3NmE0YTMyZTRlNWIuYmluZFBvcHVwKHBvcHVwX2Y1MTFhMDExYjZhNDRkZGM4ZmYyYjI4ZmUxNmJiMzdlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzg0NWRlOWU4YWVhNTQ3MThiODcxYTMxOGM1M2M3NzE4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg5NTc0MywtNzkuMzgzMTU5OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTcwOWZkNDUxNmE2NDJjY2I3NjliOWVhZGQ4NDg5MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZGU5NjhlYmE4Y2U1NDMzOWIwYTE0NWMyZGE4YmRhMWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDU0NTc4ZTk5NzMyNGJhZmI2MTA3ZjcwYWU1ZDI5NGYgPSAkKCc8ZGl2IGlkPSJodG1sXzA1NDU3OGU5OTczMjRiYWZiNjEwN2Y3MGFlNWQyOTRmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Nb29yZSBQYXJrLCBTdW1tZXJoaWxsIEVhc3QgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kZTk2OGViYThjZTU0MzM5YjBhMTQ1YzJkYThiZGExZi5zZXRDb250ZW50KGh0bWxfMDU0NTc4ZTk5NzMyNGJhZmI2MTA3ZjcwYWU1ZDI5NGYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODQ1ZGU5ZThhZWE1NDcxOGI4NzFhMzE4YzUzYzc3MTguYmluZFBvcHVwKHBvcHVwX2RlOTY4ZWJhOGNlNTQzMzliMGExNDVjMmRhOGJkYTFmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzMxNjcyOGNlNWJiNjQ0N2RiMTUzZDRhZjk3ZjA2Nzc2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg2NDEyMjk5OTk5OTksLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTcwOWZkNDUxNmE2NDJjY2I3NjliOWVhZGQ4NDg5MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjFlNWM4YjdjMzdiNDYzNmI5YzMxZmI2YTQ4ODZlMTIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOGM4NmRiY2YxYzdjNGI2NDkyZTkyZDI4NWM5NDcxN2QgPSAkKCc8ZGl2IGlkPSJodG1sXzhjODZkYmNmMWM3YzRiNjQ5MmU5MmQyODVjOTQ3MTdkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EZWVyIFBhcmssIEZvcmVzdCBIaWxsIFNFLCBSYXRobmVsbHksIFN1bW1lcmhpbGwgV2VzdCwgU291dGggSGlsbCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzIxZTVjOGI3YzM3YjQ2MzZiOWMzMWZiNmE0ODg2ZTEyLnNldENvbnRlbnQoaHRtbF84Yzg2ZGJjZjFjN2M0YjY0OTJlOTJkMjg1Yzk0NzE3ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zMTY3MjhjZTViYjY0NDdkYjE1M2Q0YWY5N2YwNjc3Ni5iaW5kUG9wdXAocG9wdXBfMjFlNWM4YjdjMzdiNDYzNmI5YzMxZmI2YTQ4ODZlMTIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTc5NGFhOGQ0MWMxNGFhODlhZTZkZWY3NWZhM2U3MjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NjI2LC03OS4zNzc1Mjk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85YjE5ZDBmMTA0M2Y0Y2NjODIzMWMzYjUxOWFjODBhOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82MDE3ZTY4YmE0ZmQ0NWM2OWM2Zjk1MTc3ZWYzMWM2NyA9ICQoJzxkaXYgaWQ9Imh0bWxfNjAxN2U2OGJhNGZkNDVjNjljNmY5NTE3N2VmMzFjNjciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvc2VkYWxlIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWIxOWQwZjEwNDNmNGNjYzgyMzFjM2I1MTlhYzgwYTguc2V0Q29udGVudChodG1sXzYwMTdlNjhiYTRmZDQ1YzY5YzZmOTUxNzdlZjMxYzY3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU3OTRhYThkNDFjMTRhYTg5YWU2ZGVmNzVmYTNlNzIxLmJpbmRQb3B1cChwb3B1cF85YjE5ZDBmMTA0M2Y0Y2NjODIzMWMzYjUxOWFjODBhOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84YjA4MTk2MDlmZjI0ZDYzYWQ2NjE0NjFkYjllY2IwMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzk2NywtNzkuMzY3Njc1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wMGYyM2Y1ZmFkYTE0ZTE4ODlkNDk4ZDcwYWU2NTA3ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zNzYxYzBmMDFmZWY0MGQxODNkZGExMzU2YWQxN2FiNyA9ICQoJzxkaXYgaWQ9Imh0bWxfMzc2MWMwZjAxZmVmNDBkMTgzZGRhMTM1NmFkMTdhYjciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNhYmJhZ2V0b3duLCBTdC4gSmFtZXMgVG93biBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzAwZjIzZjVmYWRhMTRlMTg4OWQ0OThkNzBhZTY1MDdmLnNldENvbnRlbnQoaHRtbF8zNzYxYzBmMDFmZWY0MGQxODNkZGExMzU2YWQxN2FiNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84YjA4MTk2MDlmZjI0ZDYzYWQ2NjE0NjFkYjllY2IwMC5iaW5kUG9wdXAocG9wdXBfMDBmMjNmNWZhZGExNGUxODg5ZDQ5OGQ3MGFlNjUwN2YpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjY3MDc5Y2FlOGY3NGFhZGE0YmFhZjIxY2UwOWUwZDkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjU4NTk5LC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83MDA3ZmI2MjkwMDg0YzI2YWI2M2RjOWM1YWM3MTYwNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kYmUxNGNjMWRhZjQ0OTg1OGIzYjIwMzJiZWZiZmEyZSA9ICQoJzxkaXYgaWQ9Imh0bWxfZGJlMTRjYzFkYWY0NDk4NThiM2IyMDMyYmVmYmZhMmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNodXJjaCBhbmQgV2VsbGVzbGV5IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzAwN2ZiNjI5MDA4NGMyNmFiNjNkYzljNWFjNzE2MDcuc2V0Q29udGVudChodG1sX2RiZTE0Y2MxZGFmNDQ5ODU4YjNiMjAzMmJlZmJmYTJlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzI2NzA3OWNhZThmNzRhYWRhNGJhYWYyMWNlMDllMGQ5LmJpbmRQb3B1cChwb3B1cF83MDA3ZmI2MjkwMDg0YzI2YWI2M2RjOWM1YWM3MTYwNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zODMxMTQ0NDFiZjY0Y2U5OGY5NzRjNDAzZTMyMTg0YSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1NDI1OTksLTc5LjM2MDYzNTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTcwOWZkNDUxNmE2NDJjY2I3NjliOWVhZGQ4NDg5MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZGEzOGM3MzFlYWVlNGNhY2FmZGQ3N2UzMWQ0ZThlMDcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZGVjNmU4MTU5YTg2NDI4MDg3YWJlNTBjZmVjZjU5MmYgPSAkKCc8ZGl2IGlkPSJodG1sX2RlYzZlODE1OWE4NjQyODA4N2FiZTUwY2ZlY2Y1OTJmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3VyZnJvbnQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kYTM4YzczMWVhZWU0Y2FjYWZkZDc3ZTMxZDRlOGUwNy5zZXRDb250ZW50KGh0bWxfZGVjNmU4MTU5YTg2NDI4MDg3YWJlNTBjZmVjZjU5MmYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzgzMTE0NDQxYmY2NGNlOThmOTc0YzQwM2UzMjE4NGEuYmluZFBvcHVwKHBvcHVwX2RhMzhjNzMxZWFlZTRjYWNhZmRkNzdlMzFkNGU4ZTA3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzczZTU0MjBjY2IzZjRkNWRhMmI1MDM4MzNlM2Y5ODI4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3MTYxOCwtNzkuMzc4OTM3MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTcwOWZkNDUxNmE2NDJjY2I3NjliOWVhZGQ4NDg5MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTQzNWJhZTNmNzc2NDUzODkwZjFmMzZmMjA2YzFjN2IgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzc2Y2QzMGE2ZmQ2NDZjOTk4YmM1Y2FhODZlZTRkOTggPSAkKCc8ZGl2IGlkPSJodG1sXzc3NmNkMzBhNmZkNjQ2Yzk5OGJjNWNhYTg2ZWU0ZDk4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SeWVyc29uLCBHYXJkZW4gRGlzdHJpY3QgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lNDM1YmFlM2Y3NzY0NTM4OTBmMWYzNmYyMDZjMWM3Yi5zZXRDb250ZW50KGh0bWxfNzc2Y2QzMGE2ZmQ2NDZjOTk4YmM1Y2FhODZlZTRkOTgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzNlNTQyMGNjYjNmNGQ1ZGEyYjUwMzgzM2UzZjk4MjguYmluZFBvcHVwKHBvcHVwX2U0MzViYWUzZjc3NjQ1Mzg5MGYxZjM2ZjIwNmMxYzdiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQzZmY4NzI1OGMzMDQzY2Y4N2Y3MzBkYzI5MGM2ZmQyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNDkzOSwtNzkuMzc1NDE3OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iOGRiMDJkODhkYTQ0ODRhYThiZWYyZDBmODU3ZDdmYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81N2E3ZDJjYmU3ODM0OTM3ODhjM2RhOTk1MmRjYzRmOCA9ICQoJzxkaXYgaWQ9Imh0bWxfNTdhN2QyY2JlNzgzNDkzNzg4YzNkYTk5NTJkY2M0ZjgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjhkYjAyZDg4ZGE0NDg0YWE4YmVmMmQwZjg1N2Q3ZmMuc2V0Q29udGVudChodG1sXzU3YTdkMmNiZTc4MzQ5Mzc4OGMzZGE5OTUyZGNjNGY4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQzZmY4NzI1OGMzMDQzY2Y4N2Y3MzBkYzI5MGM2ZmQyLmJpbmRQb3B1cChwb3B1cF9iOGRiMDJkODhkYTQ0ODRhYThiZWYyZDBmODU3ZDdmYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yZTY4NTBkOGYwOTc0Zjg5YjE0MmUzODYyYjk2YWJjZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NDc3MDc5OTk5OTk5NiwtNzkuMzczMzA2NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jZjg1ODY2NzM1N2E0MDZmYjY2YmRlYjY1NjU0NmU4MCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ZDY5YWFjMTBhMTY0YjU3OTI2MDA4ODFjYTUxYmM1MiA9ICQoJzxkaXYgaWQ9Imh0bWxfOWQ2OWFhYzEwYTE2NGI1NzkyNjAwODgxY2E1MWJjNTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJlcmN6eSBQYXJrIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2Y4NTg2NjczNTdhNDA2ZmI2NmJkZWI2NTY1NDZlODAuc2V0Q29udGVudChodG1sXzlkNjlhYWMxMGExNjRiNTc5MjYwMDg4MWNhNTFiYzUyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJlNjg1MGQ4ZjA5NzRmODliMTQyZTM4NjJiOTZhYmNkLmJpbmRQb3B1cChwb3B1cF9jZjg1ODY2NzM1N2E0MDZmYjY2YmRlYjY1NjU0NmU4MCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81MDcwYzQwODQ4NWM0YWY4YWM2OTUyNmU4NTI3MWEwMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1Nzk1MjQsLTc5LjM4NzM4MjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTcwOWZkNDUxNmE2NDJjY2I3NjliOWVhZGQ4NDg5MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWI2M2JhYTlkYWEyNGE3NWI3MDc2YTFlNzQ0ZmM2NDIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDEzYzQ1NWMwNjZiNDk1YmJmZGJmMzgyOWI3ZTBlYmIgPSAkKCc8ZGl2IGlkPSJodG1sXzAxM2M0NTVjMDY2YjQ5NWJiZmRiZjM4MjliN2UwZWJiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIEJheSBTdHJlZXQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lYjYzYmFhOWRhYTI0YTc1YjcwNzZhMWU3NDRmYzY0Mi5zZXRDb250ZW50KGh0bWxfMDEzYzQ1NWMwNjZiNDk1YmJmZGJmMzgyOWI3ZTBlYmIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTA3MGM0MDg0ODVjNGFmOGFjNjk1MjZlODUyNzFhMDMuYmluZFBvcHVwKHBvcHVwX2ViNjNiYWE5ZGFhMjRhNzViNzA3NmExZTc0NGZjNjQyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzljNzRjYmUyODY3ODQxY2ZiNDFkMzFjMzBlNzBhNjM4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwNTcxMjAwMDAwMDEsLTc5LjM4NDU2NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTcwOWZkNDUxNmE2NDJjY2I3NjliOWVhZGQ4NDg5MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDBjNmZiOTYyZDU5NDA4NWE3MmE1NGI3YzQwYmFmODAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTdlYjgxYzQ1YThiNDhlNDljMzhlMjJjYTY2ZDRiZWQgPSAkKCc8ZGl2IGlkPSJodG1sX2U3ZWI4MWM0NWE4YjQ4ZTQ5YzM4ZTIyY2E2NmQ0YmVkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LaW5nLCBSaWNobW9uZCwgQWRlbGFpZGUgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80MGM2ZmI5NjJkNTk0MDg1YTcyYTU0YjdjNDBiYWY4MC5zZXRDb250ZW50KGh0bWxfZTdlYjgxYzQ1YThiNDhlNDljMzhlMjJjYTY2ZDRiZWQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWM3NGNiZTI4Njc4NDFjZmI0MWQzMWMzMGU3MGE2MzguYmluZFBvcHVwKHBvcHVwXzQwYzZmYjk2MmQ1OTQwODVhNzJhNTRiN2M0MGJhZjgwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzA1OTQzNTE1ZDQ1MjRlNjJhM2U5ZmI0ZTNjZWVjM2UwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQwODE1NywtNzkuMzgxNzUyMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTcwOWZkNDUxNmE2NDJjY2I3NjliOWVhZGQ4NDg5MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzk3Njg1YWEyN2QxNGU0MGJhODMzYzczNWE0YTAwMGQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWNmYTRiNWEwNmZkNDkyYWI5YTc4Mzc3MGQ0ZTcxNjMgPSAkKCc8ZGl2IGlkPSJodG1sX2FjZmE0YjVhMDZmZDQ5MmFiOWE3ODM3NzBkNGU3MTYzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ub3JvbnRvIElzbGFuZHMsIFVuaW9uIFN0YXRpb24sIEhhcmJvdXJmcm9udCBFYXN0IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzk3Njg1YWEyN2QxNGU0MGJhODMzYzczNWE0YTAwMGQuc2V0Q29udGVudChodG1sX2FjZmE0YjVhMDZmZDQ5MmFiOWE3ODM3NzBkNGU3MTYzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzA1OTQzNTE1ZDQ1MjRlNjJhM2U5ZmI0ZTNjZWVjM2UwLmJpbmRQb3B1cChwb3B1cF8zOTc2ODVhYTI3ZDE0ZTQwYmE4MzNjNzM1YTRhMDAwZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80MzQ4N2ViYTQ2ODc0YTUzOWU5MGVjN2VhOGFjYjFlOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzE3NjgsLTc5LjM4MTU3NjQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U3MDlmZDQ1MTZhNjQyY2NiNzY5YjllYWRkODQ4OTM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzc5OWE3ZTM2NTk4NDRjMDI4MTFmY2MxNmUyNWI4NWI5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzAxMDI1MGI3ZTM1ZTQ2ODI5ZmViNzczMWNmNzlhMDk1ID0gJCgnPGRpdiBpZD0iaHRtbF8wMTAyNTBiN2UzNWU0NjgyOWZlYjc3MzFjZjc5YTA5NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VG9yb250byBEb21pbmlvbiBDZW50cmUsIERlc2lnbiBFeGNoYW5nZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzc5OWE3ZTM2NTk4NDRjMDI4MTFmY2MxNmUyNWI4NWI5LnNldENvbnRlbnQoaHRtbF8wMTAyNTBiN2UzNWU0NjgyOWZlYjc3MzFjZjc5YTA5NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80MzQ4N2ViYTQ2ODc0YTUzOWU5MGVjN2VhOGFjYjFlOC5iaW5kUG9wdXAocG9wdXBfNzk5YTdlMzY1OTg0NGMwMjgxMWZjYzE2ZTI1Yjg1YjkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzkwN2Q5Y2ViMTAxNDc2M2FhZjgwNTNiZGFhMTk3NzMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDgxOTg1LC03OS4zNzk4MTY5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMGFmYjE5NTAxZDU0N2E1YTRkNjI0Mzg2OWVjZWY5OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81ODJiMTgxZjM1NTM0MTFmOTI5YjE4NDc5NmY5NThjNyA9ICQoJzxkaXYgaWQ9Imh0bWxfNTgyYjE4MWYzNTUzNDExZjkyOWIxODQ3OTZmOTU4YzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlZpY3RvcmlhIEhvdGVsLCBDb21tZXJjZSBDb3VydCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzEwYWZiMTk1MDFkNTQ3YTVhNGQ2MjQzODY5ZWNlZjk5LnNldENvbnRlbnQoaHRtbF81ODJiMTgxZjM1NTM0MTFmOTI5YjE4NDc5NmY5NThjNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zOTA3ZDljZWIxMDE0NzYzYWFmODA1M2JkYWExOTc3My5iaW5kUG9wdXAocG9wdXBfMTBhZmIxOTUwMWQ1NDdhNWE0ZDYyNDM4NjllY2VmOTkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGVkYWY3OWM0ZDg4NDgxNzhlNmVjOTUzNzg5OGU5N2UgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTE2OTQ4LC03OS40MTY5MzU1OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80ZTA1OGJlOTE0MTQ0NGI2ODFlNDU4ZjdmM2MwZTVjYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jOTU0NDU4ODVjNDU0YWQwYjQxNmZhZTM4YmFlNTBjMiA9ICQoJzxkaXYgaWQ9Imh0bWxfYzk1NDQ1ODg1YzQ1NGFkMGI0MTZmYWUzOGJhZTUwYzIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvc2VsYXduIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGUwNThiZTkxNDE0NDRiNjgxZTQ1OGY3ZjNjMGU1Y2Muc2V0Q29udGVudChodG1sX2M5NTQ0NTg4NWM0NTRhZDBiNDE2ZmFlMzhiYWU1MGMyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RlZGFmNzljNGQ4ODQ4MTc4ZTZlYzk1Mzc4OThlOTdlLmJpbmRQb3B1cChwb3B1cF80ZTA1OGJlOTE0MTQ0NGI2ODFlNDU4ZjdmM2MwZTVjYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85YWZhZjIyMzhkMTk0N2UzYmE4YTM4YTQ3YTc3M2NlYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5Njk0NzYsLTc5LjQxMTMwNzIwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MGZmYjQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODBmZmI0IiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U3MDlmZDQ1MTZhNjQyY2NiNzY5YjllYWRkODQ4OTM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U5MzhjOTA2OWM4NjRhM2JhNWI4ZjQzNmQyZDM2OTUyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzVhNzJlYTY4ZTQ1MjQ2Y2Q4OGExZTA2ODliOGI5NTllID0gJCgnPGRpdiBpZD0iaHRtbF81YTcyZWE2OGU0NTI0NmNkODhhMWUwNjg5YjhiOTU5ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Rm9yZXN0IEhpbGwgV2VzdCwgRm9yZXN0IEhpbGwgTm9ydGggQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lOTM4YzkwNjljODY0YTNiYTViOGY0MzZkMmQzNjk1Mi5zZXRDb250ZW50KGh0bWxfNWE3MmVhNjhlNDUyNDZjZDg4YTFlMDY4OWI4Yjk1OWUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWFmYWYyMjM4ZDE5NDdlM2JhOGEzOGE0N2E3NzNjZWEuYmluZFBvcHVwKHBvcHVwX2U5MzhjOTA2OWM4NjRhM2JhNWI4ZjQzNmQyZDM2OTUyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzM4YmEwMDk1MmFhMjQzZGVhZGE3YWQzNzY4MTg5ZDBkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjcyNzA5NywtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTcwOWZkNDUxNmE2NDJjY2I3NjliOWVhZGQ4NDg5MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGRlMzM5ODI5MjcyNDUzY2FiNGJkYjZiODgwNmE4ODQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZGZlNDNkYzYzMmUzNDdkMDlhOGQwNWYwMzUzN2M5ZjQgPSAkKCc8ZGl2IGlkPSJodG1sX2RmZTQzZGM2MzJlMzQ3ZDA5YThkMDVmMDM1MzdjOWY0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aCBNaWR0b3duLCBUaGUgQW5uZXgsIFlvcmt2aWxsZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRkZTMzOTgyOTI3MjQ1M2NhYjRiZGI2Yjg4MDZhODg0LnNldENvbnRlbnQoaHRtbF9kZmU0M2RjNjMyZTM0N2QwOWE4ZDA1ZjAzNTM3YzlmNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zOGJhMDA5NTJhYTI0M2RlYWRhN2FkMzc2ODE4OWQwZC5iaW5kUG9wdXAocG9wdXBfNGRlMzM5ODI5MjcyNDUzY2FiNGJkYjZiODgwNmE4ODQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjA0NGY5MDFkNWYwNDY1ZmI1MjAyN2M5YjgyMzBhOWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI2OTU2LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U3MDlmZDQ1MTZhNjQyY2NiNzY5YjllYWRkODQ4OTM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU0NzJiN2RmOGUyOTQyYjI4ODdjOWZhY2VmNmIzOGFmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZmMmVkOWM2NGJhYTRhOTE5NjU0MWNjZDExZGZhMWVkID0gJCgnPGRpdiBpZD0iaHRtbF9mZjJlZDljNjRiYWE0YTkxOTY1NDFjY2QxMWRmYTFlZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm9yZCwgVW5pdmVyc2l0eSBvZiBUb3JvbnRvIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTQ3MmI3ZGY4ZTI5NDJiMjg4N2M5ZmFjZWY2YjM4YWYuc2V0Q29udGVudChodG1sX2ZmMmVkOWM2NGJhYTRhOTE5NjU0MWNjZDExZGZhMWVkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzIwNDRmOTAxZDVmMDQ2NWZiNTIwMjdjOWI4MjMwYTlkLmJpbmRQb3B1cChwb3B1cF81NDcyYjdkZjhlMjk0MmIyODg3YzlmYWNlZjZiMzhhZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wNjRlOWIzNGNiZWQ0YmJmOTE3ZWUyOTBlOTQyN2FlNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MzIwNTcsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTcwOWZkNDUxNmE2NDJjY2I3NjliOWVhZGQ4NDg5MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTdlNTE3YWZhMWI4NDk4MWExNGZiYzMwN2FlYjAzZTkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNGUzMGU3NWFjNzhiNDk0N2JlNWZkNWQzMjljODUwNDAgPSAkKCc8ZGl2IGlkPSJodG1sXzRlMzBlNzVhYzc4YjQ5NDdiZTVmZDVkMzI5Yzg1MDQwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HcmFuZ2UgUGFyaywgQ2hpbmF0b3duLCBLZW5zaW5ndG9uIE1hcmtldCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU3ZTUxN2FmYTFiODQ5ODFhMTRmYmMzMDdhZWIwM2U5LnNldENvbnRlbnQoaHRtbF80ZTMwZTc1YWM3OGI0OTQ3YmU1ZmQ1ZDMyOWM4NTA0MCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wNjRlOWIzNGNiZWQ0YmJmOTE3ZWUyOTBlOTQyN2FlNi5iaW5kUG9wdXAocG9wdXBfNTdlNTE3YWZhMWI4NDk4MWExNGZiYzMwN2FlYjAzZTkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjgxNTQzZDJiZGJhNDAzYTlmY2U1YjRlYjg0NjQwMzMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U3MDlmZDQ1MTZhNjQyY2NiNzY5YjllYWRkODQ4OTM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ1YWRiMjQyMWIzODRhZWJiZWE0M2Y5ZjUxMDE4YzZmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2U3Y2ExODNhM2ViODRlZTU5MWY5NmExNDZiN2U4ZjZhID0gJCgnPGRpdiBpZD0iaHRtbF9lN2NhMTgzYTNlYjg0ZWU1OTFmOTZhMTQ2YjdlOGY2YSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmF0aHVyc3QgUXVheSwgSGFyYm91cmZyb250IFdlc3QsIElzbGFuZCBhaXJwb3J0LCBDTiBUb3dlciwgU291dGggTmlhZ2FyYSwgUmFpbHdheSBMYW5kcywgS2luZyBhbmQgU3BhZGluYSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ1YWRiMjQyMWIzODRhZWJiZWE0M2Y5ZjUxMDE4YzZmLnNldENvbnRlbnQoaHRtbF9lN2NhMTgzYTNlYjg0ZWU1OTFmOTZhMTQ2YjdlOGY2YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iODE1NDNkMmJkYmE0MDNhOWZjZTViNGViODQ2NDAzMy5iaW5kUG9wdXAocG9wdXBfNDVhZGIyNDIxYjM4NGFlYmJlYTQzZjlmNTEwMThjNmYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNGVkNzg3NGFlYTA5NGJiNTk2ZGMyMTNhODY3OTIxODUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDY0MzUyLC03OS4zNzQ4NDU5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYjdiYWE1MDAxODU0MjFlODRiNWYyYzUyNTYzM2E1OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kNjhhYzRhMDJiY2U0NGQ4OWFhODE2NjI4YzQwNzJiZiA9ICQoJzxkaXYgaWQ9Imh0bWxfZDY4YWM0YTAyYmNlNDRkODlhYTgxNjYyOGM0MDcyYmYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0biBBIFBPIEJveGVzIDI1IFRoZSBFc3BsYW5hZGUgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jYjdiYWE1MDAxODU0MjFlODRiNWYyYzUyNTYzM2E1OC5zZXRDb250ZW50KGh0bWxfZDY4YWM0YTAyYmNlNDRkODlhYTgxNjYyOGM0MDcyYmYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGVkNzg3NGFlYTA5NGJiNTk2ZGMyMTNhODY3OTIxODUuYmluZFBvcHVwKHBvcHVwX2NiN2JhYTUwMDE4NTQyMWU4NGI1ZjJjNTI1NjMzYTU4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZiZTdiOGQ2YjJmZjQzN2U5YWE3MmJjNDMyNmVmOWExID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4NDI5MiwtNzkuMzgyMjgwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zYWIxMWExYTNjYTU0ZTU0YjljNmRmNjI4NDU4NTQwZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iZmM4ZDZmNWM4Zjg0ZWEyYjcxNWJmYmVlNDRlMDkzMyA9ICQoJzxkaXYgaWQ9Imh0bWxfYmZjOGQ2ZjVjOGY4NGVhMmI3MTViZmJlZTQ0ZTA5MzMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZpcnN0IENhbmFkaWFuIFBsYWNlLCBVbmRlcmdyb3VuZCBjaXR5IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2FiMTFhMWEzY2E1NGU1NGI5YzZkZjYyODQ1ODU0MGUuc2V0Q29udGVudChodG1sX2JmYzhkNmY1YzhmODRlYTJiNzE1YmZiZWU0NGUwOTMzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZiZTdiOGQ2YjJmZjQzN2U5YWE3MmJjNDMyNmVmOWExLmJpbmRQb3B1cChwb3B1cF8zYWIxMWExYTNjYTU0ZTU0YjljNmRmNjI4NDU4NTQwZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lNzAwYTVhY2YyMjE0MmZkOTI1NzgyZTg1OTgxZGFhYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTU0MiwtNzkuNDIyNTYzN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81ZTUyOTVjZjBlN2U0NjU2YTdjNmQzMTIzN2ZmZTRjZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kYWVhNjBmM2ViMzQ0YWE4ODdkNjVmOGQ2ZTkzYTgyMCA9ICQoJzxkaXYgaWQ9Imh0bWxfZGFlYTYwZjNlYjM0NGFhODg3ZDY1ZjhkNmU5M2E4MjAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNocmlzdGllIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNWU1Mjk1Y2YwZTdlNDY1NmE3YzZkMzEyMzdmZmU0Y2Quc2V0Q29udGVudChodG1sX2RhZWE2MGYzZWIzNDRhYTg4N2Q2NWY4ZDZlOTNhODIwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U3MDBhNWFjZjIyMTQyZmQ5MjU3ODJlODU5ODFkYWFiLmJpbmRQb3B1cChwb3B1cF81ZTUyOTVjZjBlN2U0NjU2YTdjNmQzMTIzN2ZmZTRjZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kYTdiYzQ2MDA4Mzk0NmQ0OTU3MWRiNmM0ZTkyMTY3NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTAwNTEwMDAwMDAxLC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U3MDlmZDQ1MTZhNjQyY2NiNzY5YjllYWRkODQ4OTM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNlN2ZhZThmZTkzOTRhNzY4MGQ0ZWM3OGJiZmYxYmRiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZlNjc1MmJhMjAwZDRiNzBhMWEwYzA4ZmZkOTIyMDYzID0gJCgnPGRpdiBpZD0iaHRtbF82ZTY3NTJiYTIwMGQ0YjcwYTFhMGMwOGZmZDkyMjA2MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG92ZXJjb3VydCBWaWxsYWdlLCBEdWZmZXJpbiBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzNlN2ZhZThmZTkzOTRhNzY4MGQ0ZWM3OGJiZmYxYmRiLnNldENvbnRlbnQoaHRtbF82ZTY3NTJiYTIwMGQ0YjcwYTFhMGMwOGZmZDkyMjA2Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kYTdiYzQ2MDA4Mzk0NmQ0OTU3MWRiNmM0ZTkyMTY3Ny5iaW5kUG9wdXAocG9wdXBfM2U3ZmFlOGZlOTM5NGE3NjgwZDRlYzc4YmJmZjFiZGIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzExMDgxNzdhZDY5NDU1NTkzOGE5MDQ0YmM0MjUxYzMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDc5MjY3MDAwMDAwMDYsLTc5LjQxOTc0OTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTcwOWZkNDUxNmE2NDJjY2I3NjliOWVhZGQ4NDg5MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOGU4ZTg3MjM2Y2MyNDgyZTk3YWRlNzA2MWI4ODZmNDMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTI3MGE1NGM1YWIyNDJlYTllZjNkZDZiYjkxOWVjYjQgPSAkKCc8ZGl2IGlkPSJodG1sXzUyNzBhNTRjNWFiMjQyZWE5ZWYzZGQ2YmI5MTllY2I0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UcmluaXR5LCBMaXR0bGUgUG9ydHVnYWwgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84ZThlODcyMzZjYzI0ODJlOTdhZGU3MDYxYjg4NmY0My5zZXRDb250ZW50KGh0bWxfNTI3MGE1NGM1YWIyNDJlYTllZjNkZDZiYjkxOWVjYjQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzExMDgxNzdhZDY5NDU1NTkzOGE5MDQ0YmM0MjUxYzMuYmluZFBvcHVwKHBvcHVwXzhlOGU4NzIzNmNjMjQ4MmU5N2FkZTcwNjFiODg2ZjQzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE5OGZjYWI2M2I2ZTQxODZiYzA0MDdlNGU4NTExMTcyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjM2ODQ3MiwtNzkuNDI4MTkxNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTcwOWZkNDUxNmE2NDJjY2I3NjliOWVhZGQ4NDg5MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWVmZTI3OTEwN2RiNDVmZmEzNTIwNDU2N2MzN2M4MWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfM2M1MTRmZDA5MzA3NGMwYmIyZGI4YzQ2YmQ5MWRiZDUgPSAkKCc8ZGl2IGlkPSJodG1sXzNjNTE0ZmQwOTMwNzRjMGJiMmRiOGM0NmJkOTFkYmQ1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYXJrZGFsZSBWaWxsYWdlLCBCcm9ja3RvbiwgRXhoaWJpdGlvbiBQbGFjZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FlZmUyNzkxMDdkYjQ1ZmZhMzUyMDQ1NjdjMzdjODFmLnNldENvbnRlbnQoaHRtbF8zYzUxNGZkMDkzMDc0YzBiYjJkYjhjNDZiZDkxZGJkNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xOThmY2FiNjNiNmU0MTg2YmMwNDA3ZTRlODUxMTE3Mi5iaW5kUG9wdXAocG9wdXBfYWVmZTI3OTEwN2RiNDVmZmEzNTIwNDU2N2MzN2M4MWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjE2OTg0ZjJkMzU4NDMwN2ExYjgzOTc5ZThhNDc5ZWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjE2MDgzLC03OS40NjQ3NjMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hYTkyZTMxNjc0NDc0ODFkOTJlYmMzMDUyNjc5NWJjNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iYzk2NTliYzNhMmQ0MDQ3YTNlYTBlNzI1YmIyZDk4NyA9ICQoJzxkaXYgaWQ9Imh0bWxfYmM5NjU5YmMzYTJkNDA0N2EzZWEwZTcyNWJiMmQ5ODciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBKdW5jdGlvbiBTb3V0aCwgSGlnaCBQYXJrIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYWE5MmUzMTY3NDQ3NDgxZDkyZWJjMzA1MjY3OTViYzYuc2V0Q29udGVudChodG1sX2JjOTY1OWJjM2EyZDQwNDdhM2VhMGU3MjViYjJkOTg3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzYxNjk4NGYyZDM1ODQzMDdhMWI4Mzk3OWU4YTQ3OWVhLmJpbmRQb3B1cChwb3B1cF9hYTkyZTMxNjc0NDc0ODFkOTJlYmMzMDUyNjc5NWJjNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kYzA1MTlhZjg0MzE0YTk2YmI1ZTA2YTM2ZWE3YjIwOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODk1OTcsLTc5LjQ1NjMyNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82MzVlODM0MzYyOTM0MTY1YWNlZmI3NzcyMDEzZDQwOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lOThlMjQ2MzVmNDM0OGM2YmY1YzhkZTE0NzE0ZWY3NCA9ICQoJzxkaXYgaWQ9Imh0bWxfZTk4ZTI0NjM1ZjQzNDhjNmJmNWM4ZGUxNDcxNGVmNzQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBhcmtkYWxlLCBSb25jZXN2YWxsZXMgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82MzVlODM0MzYyOTM0MTY1YWNlZmI3NzcyMDEzZDQwOS5zZXRDb250ZW50KGh0bWxfZTk4ZTI0NjM1ZjQzNDhjNmJmNWM4ZGUxNDcxNGVmNzQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZGMwNTE5YWY4NDMxNGE5NmJiNWUwNmEzNmVhN2IyMDkuYmluZFBvcHVwKHBvcHVwXzYzNWU4MzQzNjI5MzQxNjVhY2VmYjc3NzIwMTNkNDA5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ4N2Q1ZDhkMDhlYTQyM2ZiYmFiZjk2YjZkYzY1YzRjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNTcwNiwtNzkuNDg0NDQ5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNzA5ZmQ0NTE2YTY0MmNjYjc2OWI5ZWFkZDg0ODkzOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jODFiNDBkZjBiYTk0ZDQ1OTdmMGJkNTkxNGYxZDBmMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NDMwNmUzZDg2ZjE0ZTg1OGM1OTQ3NDE4YjgyMmVmYyA9ICQoJzxkaXYgaWQ9Imh0bWxfNDQzMDZlM2Q4NmYxNGU4NThjNTk0NzQxOGI4MjJlZmMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN3YW5zZWEsIFJ1bm55bWVkZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M4MWI0MGRmMGJhOTRkNDU5N2YwYmQ1OTE0ZjFkMGYxLnNldENvbnRlbnQoaHRtbF80NDMwNmUzZDg2ZjE0ZTg1OGM1OTQ3NDE4YjgyMmVmYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80ODdkNWQ4ZDA4ZWE0MjNmYmJhYmY5NmI2ZGM2NWM0Yy5iaW5kUG9wdXAocG9wdXBfYzgxYjQwZGYwYmE5NGQ0NTk3ZjBiZDU5MTRmMWQwZjEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzVhNjcwNWFlOGY3NDYyNGI4ZDNlNmZlMDkzYzNlZDIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI3NDM5LC03OS4zMjE1NThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTcwOWZkNDUxNmE2NDJjY2I3NjliOWVhZGQ4NDg5MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWRlZGRjZWQzOGYyNDBiNGJiYjAzNWI5MjlkNDcyNmMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjRiYzNjMGFmNmIyNDk3NTk0NTJlZDdhOTY3NTg3YmYgPSAkKCc8ZGl2IGlkPSJodG1sX2Y0YmMzYzBhZjZiMjQ5NzU5NDUyZWQ3YTk2NzU4N2JmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CdXNpbmVzcyBSZXBseSBNYWlsIFByb2Nlc3NpbmcgQ2VudHJlIDk2OSBFYXN0ZXJuIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZWRlZGRjZWQzOGYyNDBiNGJiYjAzNWI5MjlkNDcyNmMuc2V0Q29udGVudChodG1sX2Y0YmMzYzBhZjZiMjQ5NzU5NDUyZWQ3YTk2NzU4N2JmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzM1YTY3MDVhZThmNzQ2MjRiOGQzZTZmZTA5M2MzZWQyLmJpbmRQb3B1cChwb3B1cF9lZGVkZGNlZDM4ZjI0MGI0YmJiMDM1YjkyOWQ0NzI2Yyk7CgogICAgICAgICAgICAKICAgICAgICAKPC9zY3JpcHQ+" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



<a id='item5'></a>
