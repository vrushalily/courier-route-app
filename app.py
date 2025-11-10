# =========================================================
# 📦 Courier Route Optimization (Streamlit Web App)
# =========================================================
import streamlit as st
import folium
import openrouteservice
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from streamlit_folium import st_folium
import random

# ========= CONFIG ==========
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImVmNzM5YTUyYzUzMDQ0NTJhZDkyYzFjZTE4ODM0Y2Q4IiwiaCI6Im11cm11cjY0In0="

# ========= FUNCTIONS ==========
def geocode_locations(location_names):
    geolocator = Nominatim(user_agent="courier_optimizer")
    coords = []
    for name in location_names:
        loc = geolocator.geocode(name)
        if loc:
            coords.append((loc.latitude, loc.longitude))
        else:
            st.warning(f"❌ Could not find location: {name}")
    return coords

def update_dist_with_traffic(dist, traffic_level):
    n = len(dist)
    return [[dist[i][j] * (1 + random.uniform(0, traffic_level)) for j in range(n)] for i in range(n)]

def total_cost(dist, i, j, fuel_efficiency, fuel_price, toll_costs, weather_penalty):
    fuel_cost = (dist[i][j] / fuel_efficiency) * fuel_price
    cost = fuel_cost + toll_costs[i][j]
    cost *= (1 + weather_penalty[i][j])
    return cost

def tsp_dp(dist, time_windows, service_time, priority,
           fuel_efficiency, fuel_price, toll_costs, weather_penalty):
    n = len(dist)
    dp = [[float('inf')] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    time_arrival = [[0.0] * n for _ in range(1 << n)]
    dp[1][0] = 0.0

    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                travel_cost = total_cost(dist, u, v, fuel_efficiency, fuel_price, toll_costs, weather_penalty)
                new_cost = dp[mask][u] + travel_cost / max(priority[v], 1)
                arrival_time = max(time_arrival[mask][u] + dist[u][v], time_windows[v][0])
                if arrival_time <= time_windows[v][1] and new_cost < dp[new_mask][v]:
                    dp[new_mask][v] = new_cost
                    parent[new_mask][v] = u
                    time_arrival[new_mask][v] = arrival_time + service_time[v]

    end_mask = (1 << n) - 1
    min_cost, last = float('inf'), -1
    for i in range(n):
        cost = dp[end_mask][i] + total_cost(dist, i, 0, fuel_efficiency, fuel_price, toll_costs, weather_penalty)
        if cost < min_cost:
            min_cost, last = cost, i

    path = []
    mask = end_mask
    while last != -1:
        path.append(last)
        tmp = parent[mask][last]
        mask &= ~(1 << last)
        last = tmp
    path.append(0)
    path.reverse()
    return path, min_cost

def ors_matrix(coords_lonlat, key):
    client = openrouteservice.Client(key=key)
    res = client.distance_matrix(locations=coords_lonlat, profile="driving-car",
                                 metrics=["distance", "duration"], validate=True)
    dist_km = [[d / 1000 for d in row] for row in res["distances"]]
    dur_s = res["durations"]
    return dist_km, dur_s

def fallback_matrix(coords_latlon):
    n = len(coords_latlon)
    dist = [[0] * n for _ in range(n)]
    dur = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            km = geodesic(coords_latlon[i], coords_latlon[j]).km
            dist[i][j] = km
            dur[i][j] = km / 35.0 * 3600
    return dist, dur

def draw_route_map(points, route, ors_key=None):
    m = folium.Map(location=points[0], zoom_start=12)
    for i, (lat, lon) in enumerate(points):
        folium.Marker(
            [lat, lon],
            popup=f"{i}: Stop",
            icon=folium.Icon(color="green" if i == 0 else "blue")
        ).add_to(m)
    client = None
    if ors_key:
        try:
            client = openrouteservice.Client(key=ors_key)
        except:
            pass
    for a, b in zip(route[:-1], route[1:]):
        if client:
            try:
                geo = client.directions([(points[a][1], points[a][0]),
                                         (points[b][1], points[b][0])],
                                        profile="driving-car", format="geojson")
                folium.GeoJson(geo,
                               style_function=lambda x: {"color": "red", "weight": 5}).add_to(m)
                continue
            except:
                pass
        folium.PolyLine([[points[a][0], points[a][1]], [points[b][0], points[b][1]]],
                        color="red", weight=5, opacity=0.7).add_to(m)
    return m

# ========= STREAMLIT UI ==========
st.title("📦 Courier Route Optimization by Location")
st.write("Enter delivery points (including starting depot):")

num_points = st.number_input("Number of points:", min_value=2, max_value=10, value=3)
loc_names = [st.text_input(f"Location {i+1} name:", "") for i in range(num_points)]

fuel_price = st.number_input("Fuel price (₹/L):", min_value=50.0, value=100.0)
fuel_eff = st.number_input("Fuel efficiency (km/L):", min_value=1.0, value=15.0)
traffic = st.slider("Traffic level:", 0.0, 1.0, 0.2)

if st.button("🚀 Optimize Route"):
    if all(loc_names):
        points = geocode_locations(loc_names)
        if len(points) == num_points:
            try:
                coords_lonlat = [(p[1], p[0]) for p in points]
                dist, dur = ors_matrix(coords_lonlat, ORS_API_KEY)
            except:
                st.warning("⚠ Using fallback geodesic distances")
                dist, dur = fallback_matrix(points)

            dist = update_dist_with_traffic(dist, traffic)
            n = len(dist)
            time_windows = [(0, 1e9)] * n
            service_time = [0] * n
            priority = [1] * n
            toll_costs = [[0] * n for _ in range(n)]
            weather = [[0] * n for _ in range(n)]

            route, cost = tsp_dp(dist, time_windows, service_time, priority,
                                 fuel_eff, fuel_price, toll_costs, weather)
            total_km = sum(dist[a][b] for a, b in zip(route[:-1], route[1:]))
            total_min = sum(dur[a][b] for a, b in zip(route[:-1], route[1:])) / 60

            st.success(f"✅ Optimized Route: {' → '.join(loc_names[i] for i in route)}")
            st.write(f"**Total Distance:** {total_km:.2f} km")
            st.write(f"**Estimated Time:** {total_min:.1f} minutes")
            st.write(f"**Total Cost:** ₹{cost:.2f}")

            st_map = draw_route_map(points, route, ors_key=ORS_API_KEY)
            st_folium(st_map, width=700, height=500)
    else:
        st.warning("Please enter all location names.")
