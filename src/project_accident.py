import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd

if __name__ == "__main__":
    # Read the CSV file into a Pandas DataFrame
    incident_df = pd.read_csv("processed_data/2023_incidents.csv")

    # Create a GeoDataFrame with Point geometries
    incident_gdf = gpd.GeoDataFrame(
        incident_df,
        geometry=gpd.points_from_xy(incident_df.longitude, incident_df.latitude),
        crs="EPSG:4326",
    )
    incident_utm = incident_gdf.to_crs(32647)

    # Bound inside bangkok
    bangkok = ox.geocode_to_gdf("R92277", by_osmid=True)
    bangkok_utm = bangkok.to_crs(32647)
    incident_utm = incident_utm[incident_utm.intersects(bangkok_utm.geometry.iloc[0])]
    # print(incident_utm.geometry.x)
    # print(incident_utm.geometry.y)

    # Load road network
    G = ox.graph_from_place("Bangkok, Thailand", network_type="drive")
    G = ox.project_graph(G, to_crs=32647)

    # Find closest node
    incident_nodes = ox.distance.nearest_nodes(
        G, incident_utm.geometry.x, incident_utm.geometry.y
    )
    road_nodes, road_edges = ox.graph_to_gdfs(G)
    # print(incident_nodes)
    # print(road_nodes)

    ax = bangkok_utm.plot(color="white", edgecolor="black")
    ax = road_nodes.loc[incident_nodes].plot(
        ax=ax, marker="o", color="red", markersize=5, zorder=5
    )
    fig, ax = ox.plot_graph(G, ax=ax, node_zorder=0)
    plt.show()
