# Public Transportation Usage – Trends & Visualization 

This project was developed as part of the **Information Visualization** course.  
We analyze **public transportation usage patterns in Israel (2020–2025)** and present them through **interactive visual dashboards**, focusing on temporal and spatial trends.

The goal is to explore how public transportation usage changes over time, across cities, stations, and different hours of the day, and to enable intuitive data exploration via visualization.

---


## Data Sources

The data was obtained from the official Israeli open data portal:

- **Public Transportation Usage (Ticket Validations by Station)**  
  https://data.gov.il/he/datasets/ministry_of_transport/tikufim_station_2022

- **Bus Stops Metadata (Locations & Attributes)**  
  https://data.gov.il/he/datasets/ministry_of_transport/bus_stops

All datasets are published under an open data license.

---

## Project Overview

- Data cleaning and preprocessing (large-scale CSV files)
- Aggregation of rides by station, city, time, and peak/off-peak hours
- Interactive visualizations (maps, time series, comparisons)

---

## Repository Structure

- `Create Dataset/` – data cleaning and preprocessing scripts
- `Dashboard/` – visualization notebooks and dashboard logic  

---

## Notes

Due to the size of the datasets (2.81 GB), **raw and cleaned data files are not included** in the repository and must be downloaded separately from the sources above.
