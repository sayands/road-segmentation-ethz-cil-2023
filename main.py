import requests
from PIL import Image
from io import BytesIO
import numpy as np
import os

KEY = os.getenv('GMAPS_KEY')
assert KEY
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
        -74.2591, # West
        -73.7004, # East
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
    ], [
        # Zurich, Switzerland
        47.3203031, # South
        47.4308514, # North
        8.4487451, # West
        8.5977255, # East
    ], [
        # Sofia, Bulgaria
        42.6039868, # South
        42.7831884, # North
        23.2116106, # West
        23.4761724, # East
    ], [
        # Hyderabad, India
        17.2190976, # South
        17.5950484, # North
        78.1633353, # West
        78.6518683, # East
    ]
]


CROP_PIXELS =  20.
MIN_ROAD_PERCENTAGE = 10.
DATA_FOLDER = "data"

ZOOM_LEVEL = 15
ITEARTION_STEP = .035


def get_satelite_image(location, folder, id):
    center_x, center_y = location
    url = "https://maps.googleapis.com/maps/api/staticmap?"+\
        f"center={center_x},{center_y}"+\
        f"&zoom={ZOOM_LEVEL}&size=1020x1020"+\
        f"&format=PNG&maptype=satellite&key={KEY}"
    response = requests.get(url)

    image = Image.open(BytesIO(response.content))
    # Get image size
    width, height = image.size

    # Crop image to remove bottom 20 pixels
    cropped_image = image.crop((0, 0, width - CROP_PIXELS, height - CROP_PIXELS))
    cropped_image.convert("RGB").save(f"{DATA_FOLDER}/{folder}/{id}.png")


def get_street_labels(location, folder, id):
    center_x, center_y = location
    url = "https://maps.googleapis.com/maps/api/staticmap?" +\
        f"center={center_x},{center_y}&key={KEY}&zoom={ZOOM_LEVEL}&"+\
        "size=1020x1020&maptype=roadmap&format=PNG&"+\
        "style=feature:all|element:labels|visibility:off&"+\
        "style=feature:administrative|visibility:off&"+\
        "style=feature:landscape|visibility:off&"+\
        "style=feature:poi|visibility:off&"+\
        "style=feature:water|visibility:off&"+\
        "style=feature:transit|visibility:off&"+\
        "style=feature:road|element:geometry|color:0xffffff"

    response = requests.get(url)
    img_arr = np.array(Image.open(BytesIO(response.content)))
    img_arr[img_arr != 0] = 255

    image = Image.fromarray(img_arr)

    width, height = image.size

    # Crop image to remove bottom 20 pixels
    cropped_image = image.crop((0, 0, width - CROP_PIXELS, height - CROP_PIXELS))

    label_percent = np.count_nonzero(img_arr) * 100.0 / ((width - CROP_PIXELS) * (height - CROP_PIXELS))
    if label_percent > MIN_ROAD_PERCENTAGE:
        if not os.path.exists(f"{DATA_FOLDER}/{folder}"):
            os.makedirs(f"{DATA_FOLDER}/{folder}")
        cropped_image.convert("RGB").save(f"{DATA_FOLDER}/{folder}/{id}_label.png")
    else:
        print("Skipped with ", label_percent)
    return label_percent


def get_data(locations):
    location_folder_id = 1
    for location in locations:
        location_folder = f"{location_folder_id}_ZOOM_{ZOOM_LEVEL}"
        file_id = 1

        # By test and trial I got these number as full move in each direction without intersections between images
        # For zoom 16 - .035 
        # For zoom 15 - 0.07
        step_x = .035 if ZOOM_LEVEL == 16 else .07 # TODO: Make configs
        step_y = .035 if ZOOM_LEVEL == 16 else .07 # TODO: Make configs
        start_x, end_x, start_y, end_y = location

        num_of_scans = int((abs(start_x - end_x) / step_x) * (abs(start_y - end_y) / step_y))

        print(f"Starting scanning for location {location_folder}.")
        print(f"Scanned data will be: {num_of_scans} files.")

        while start_x <= end_x:
            cur_y = start_y
            while cur_y <= end_y:
                label_percent = get_street_labels((start_x, cur_y), location_folder, file_id)
                if label_percent > MIN_ROAD_PERCENTAGE:
                    get_satelite_image((start_x, cur_y), location_folder, file_id)
                    file_id += 1
                cur_y += step_y
                if file_id % 10 == 0:
                    print(f"Scanned through {file_id} files. For location {location_folder}")
            start_x += step_x
        location_folder_id += 1
    print("Collected all data.")

get_data(city_coordinates)
ZOOM_LEVEL +=1 
get_data(city_coordinates)