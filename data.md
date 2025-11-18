# Data
We have the following data from iTIC Foundation:
## 1. Traffic Accidents
iTIC records traffic incidents from an RSS-style feed in 30-minute increments. Each 30-minute block of incidents is saved as an .xml file. 
To do:
- Loop over each .xml file, extract contents into .csv
- We want 1 .csv file for each year (2018,...,2023).
- Each .csv file should contain accident data with the following info:
  1. Latitude
  2. Longitude
  3. Start time (occurence)
  4. Road name (English)
  5. Description (English)

## 2. Congestion
iTIC records congestion data by assigning _congestion levels_ to road segments between pairs of Locations. Each Location is a pre-defined Traffic Management Channel (TMC) point.
The full list of TMC points can be found at https://traffic.longdo.com/opendata/location-table/v3.2/.
This congestion data implicitly defines a graph $(T, E)$ where $T$ is the set of TMC points in Bangkok and $E$ is the set of edges between these points, i.e. the road network.

# Forming the Bangkok Road Graph
Let $(T,E)$ be the graph defined by the congestion data. We will augment this graph for the purpose of incorporating ambulance depot and accident locations. 
First, it is useful to define the following workflow:
```
def add_nodes_to_graph(set S, graph (V, E)):
  for location i in S:
    # 1. find nearest edge and project
    e = nearest_edge(i, E)
    i' = proj(i, e)
  
    # 2. add projected point to graph
    add i' to V
  
    # 3. split associated edge(s)
    x, y = endpoints(e)
    add (x, i') and (i', y) to E
    remove (x, y) from E
    if reverse direction (y, x) in E:
      add (y, i') and (i', x) to E
      remove (y, x) from E

  return (V, E)
```

## 1. Depot Locations
We consider the following candidate locations:
1. Hospitals
2. Fire Stations
3. Police Stations
4. Government Buildings
5. Shopping Malls
6. Train Stations

```
D = get_from_OSM(Bangkok, depot filter)
add_nodes_to_graph(D, (T, E))
```

## 2. Accident Locations
```
given set of accident points A
Ak = k_means_clustering(A, k)
add_nodes_to_graph(Ak, (T, E))
```
