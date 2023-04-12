import requests

key = "AIzaSyBRr3cZlbyDizvYzRUJvVirv2-B5PMyj_0"

city_coordinates = [[
        # Tokyo, Japan
        35.5536, # South
        35.8174, # North
        139.2392, # West
        139.9289, # East
    ],[
        # New York City, USA
        40.4961, # South
        40.9176, # North
        -73.7004, # West
        -74.2591, # East
    ],[
        # Sao Paulo, Brazil
        -23.8162, # South
        -23.0016, # North
        -47.1429, # West
        -46.3650, # East
    ],[
        # Moscow, Russia
        55.4909, # Sourth
        56.0094, # North
        36.8030, # West
        37.9664 # East
    ],[
        # Paris, France
        48.8156, # South
        49.0047, # North
        2.2241, # West
        2.4699, # East
]]


ZOOM_LEVEL = 16

def get_satelite_image(location, folder, id):
    center_x, center_y = location
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={center_x},{center_y}&zoom={ZOOM_LEVEL}&size=1024x1024&format=JPEG&maptype=satellite&key={key}"
    response = requests.get(url)

    with open(f"data/{folder}/{id}.jpg", "wb") as f:
        f.write(response.content)


def get_street_label(location, folder, id):
    center_x, center_y = location
    url = f"https://maps.googleapis.com/maps/api/staticmap?" +\
      f"center={center_x},{center_y}&&"+\
        f"zoom={ZOOM_LEVEL}&size=1024x1024&maptype=roadmap&format=JPEG&"+\
        "style=feature:all|element:labels|visibility:off&"+\
        "style=feature:administrative|visibility:off&"+\
        "style=feature:landscape|visibility:off&"+\
        "style=feature:poi|visibility:off&"+\
        "style=feature:water|visibility:off&"+\
        "style=feature:transit|visibility:off&"+\
        "style=feature:road|element:geometry|color:0xffffff"+\
        f"&key={key}"
    response = requests.get(url)
    with open(f"data/{folder}/{id}_label.jpg", "wb") as f:
        f.write(response.content)

def get_data(locations):
    location_folder = 6
    for location in locations:
        # By test and trial I got to .15 as full move in each direction without having the images intersect.
        step_x = .035
        step_y = .035
        
        file_id = 1
        
        start_x, end_x, start_y, end_y = location
        print(start_x, end_x, start_y, end_y)
        num_of_scans = int((abs(start_x - end_x) / step_x) * (abs(start_y - end_y) / step_y))

        print(f"Starting scanning for location {location_folder}.")
        print(f"Scanned data will be: {num_of_scans} files.")

        while start_x <= end_x:
            cur_y = start_y
            while cur_y <= end_y:
                get_satelite_image((start_x, cur_y), location_folder, file_id)
                get_street_label((start_x, cur_y), location_folder, file_id)
                cur_y += step_y
                file_id += 1
                if file_id % 100:
                    print(f"Scanned through {file_id} files. For location {location_folder}")
            start_x += step_x
        location_folder += 1
    print("Collected all data.")

# 6 and on are with zoom level 16

get_data(city_coordinates)