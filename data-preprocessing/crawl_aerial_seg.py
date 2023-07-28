import requests
from PIL import Image
from io import BytesIO
import numpy as np
import os

KEYS = ["AIzaSyByICrMoOw9O2APN5b9S0lEcIqLw1vSnt4", "AIzaSyBRr3cZlbyDizvYzRUJvVirv2-B5PMyj_0",
        "AIzaSyDMv_Ve7nOQKlEorJWjX3GmfPeg7hC06DY", "AIzaSyC50PF5uy5nIVlrH4wc1Hkk3pJB79AypAo"]  # os.getenv('GMAPS_KEY')
key = KEYS[2]
assert key

city_coordinates = [
    [
        # Buenos Aires, Argentina
        -34.9214,  # South
        -34.4607,  # North
        -58.5308,  # West
        -58.1307,  # East
    ],
    [
        # Sucre (Official), La Paz (Seat of Government), Bolivia
        -17.5285,  # South
        -13.0472,  # North
        -69.9597,  # West
        -63.1965,  # East
    ],
    [
        # Brasilia, Brazil
        -15.9899,  # South
        -15.4113,  # North
        -48.2353,  # West
        -47.6866,  # East
    ],
    [
        # Santiago, Chile
        -33.5912,  # South
        -33.0153,  # North
        -71.0179,  # West
        -70.4477,  # East
    ],
    [
        # Bogota, Colombia
        4.1990,  # South
        5.2003,  # North
        -74.2299,  # West
        -73.9862,  # East
    ],
    [
        # Quito, Ecuador
        -0.5475,  # South
        -0.0037,  # North
        -78.6115,  # West
        -78.2784,  # East
    ],
    [
        # Asuncion, Paraguay
        -25.6086,  # South
        -25.1557,  # North
        -57.7542,  # West
        -57.4663,  # East
    ],
    [
        # Lima, Peru
        -12.2404,  # South
        -11.7332,  # North
        -77.1825,  # West
        -76.6997,  # East
    ],
    [
        # Montevideo, Uruguay
        -34.9730,  # South
        -34.7246,  # North
        -56.4152,  # West
        -56.0269,  # East
    ],
    [
        # Georgetown, Guyana
        6.7242,  # South
        6.8046,  # North
        -58.2020,  # West
        -58.1211,  # East
    ],
    [
        # Paramaribo, Suriname
        5.7700,  # South
        6.0043,  # North
        -55.1986,  # West
        -54.9949,  # East
    ],
    [
        # Caracas, Venezuela
        10.0670,  # South
        10.5720,  # North
        -67.0311,  # West
        -66.8019,  # East
    ],
    [
        # Shanghai, China
        30.9756,  # South
        31.5149,  # North
        120.8604,  # West
        122.1191,  # East
    ],
    [
        # Delhi, India
        28.4022,  # South
        28.8835,  # North
        76.8381,  # West
        77.3475,  # East
    ],
    [
        # Karachi, Pakistan
        24.7294,  # South
        25.1719,  # North
        66.3230,  # West
        67.1829,  # East
    ],
    [
        # Tehran, Iran
        35.5049,  # South
        35.8507,  # North
        51.1839,  # West
        51.6867,  # East
    ],
    [
        # Riyadh, Saudi Arabia
        24.4682,  # South
        24.9735,  # North
        46.5068,  # West
        47.0142,  # East
    ],
    [
        # Manila, Philippines
        14.4000,  # South
        14.8369,  # North
        120.8588,  # West
        121.0843,  # East
    ],
    [
        # Jakarta, Indonesia
        -6.4242,  # South
        -5.8899,  # North
        106.5761,  # West
        107.1185,  # East
    ],
    [
        # Osaka, Japan
        34.4907,  # South
        34.8718,  # North
        135.3315,  # West
        135.8400,  # East
    ],
    [
        # Madrid, Spain
        40.3173,  # South
        40.6436,  # North
        -3.8770,  # West
        -3.5174,  # East
    ],
    [
        # Vienna, Austria
        48.1104,  # South
        48.3204,  # North
        16.1867,  # West
        16.5775,  # East
    ],
    [
        # Budapest, Hungary
        47.4020,  # South
        47.5715,  # North
        18.9034,  # West
        19.1627,  # East
    ],
    [
        # Amsterdam, Netherlands
        52.2893,  # South
        52.4310,  # North
        4.7281,  # West
        5.0794,  # East
    ],
    [
        # Prague, Czech Republic
        49.9464,  # South
        50.1765,  # North
        14.2251,  # West
        14.7066,  # East
    ],
    [
        # Warsaw, Poland
        52.0909,  # South
        52.3676,  # North
        20.8425,  # West
        21.2710,  # East
    ],
    [
        # Stockholm, Sweden
        59.1959,  # South
        59.4376,  # North
        17.0983,  # West
        18.1708,  # East
    ],
    [
        # Copenhagen, Denmark
        55.6174,  # South
        55.7331,  # North
        12.3390,  # West
        12.6501,  # East
    ],
    [
        # Athens, Greece
        37.9413,  # South
        38.0796,  # North
        23.6370,  # West
        23.8981,  # East
    ],
    [
        # Dublin, Ireland
        53.2048,  # South
        53.4251,  # North
        -6.3872,  # West
        -6.1077,  # East
    ],
    [
        # Brussels, Belgium
        50.7550,  # South
        50.9131,  # North
        4.2376,  # West
        4.4858,  # East
    ],
    [
        # Oslo, Norway
        59.8240,  # South
        60.0027,  # North
        10.6693,  # West
        11.0335,  # East
    ],
    [
        # Lisbon, Portugal
        38.6603,  # South
        38.7990,  # North
        -9.2278,  # West
        -9.1103,  # East
    ],
    [
        # Helsinki, Finland
        60.1254,  # South
        60.3200,  # North
        24.7829,  # West
        25.2254,  # East
    ],
    [
        # Reykjavik, Iceland
        64.0401,  # South
        64.1863,  # North
        -22.0830,  # West
        -21.6947,  # East
    ],
    [
        # Alabama - Birmingham
        33.2479,  # South
        33.6786,  # North
        -87.2962,  # West
        -86.5795,  # East
    ],
    [
        # Alaska - Anchorage
        61.1008,  # South
        61.4327,  # North
        -150.0847,  # West
        -149.5612,  # East
    ],
    [
        # Arizona - Phoenix
        33.2903,  # South
        33.9204,  # North
        -112.3246,  # West
        -111.7897,  # East
    ],
    [
        # Arkansas - Little Rock
        34.5794,  # South
        34.8743,  # North
        -92.6194,  # West
        -92.1253,  # East
    ],
    [
        # Colorado - Denver
        39.6144,  # South
        39.9142,  # North
        -105.1099,  # West
        -104.6003,  # East
    ],
    [
        # Connecticut - Bridgeport
        41.1145,  # South
        41.2434,  # North
        -73.2387,  # West
        -73.1609,  # East
    ],
    [
        # Delaware - Wilmington
        39.6782,  # South
        39.8396,  # North
        -75.6057,  # West
        -75.4105,  # East
    ],
    [
        # Florida - Jacksonville
        30.1905,  # South
        30.4770,  # North
        -81.7884,  # West
        -81.3961,  # East
    ],
    [
        # Georgia - Atlanta
        33.5024,  # South
        33.8871,  # North
        -84.5766,  # West
        -84.2890,  # East
    ],
    [
        # Hawaii - Honolulu
        21.2541,  # South
        21.5014,  # North
        -157.9734,  # West
        -157.6478,  # East
    ],
    [
        # Idaho - Boise
        43.4599,  # South
        43.7137,  # North
        -116.3446,  # West
        -116.0735,  # East
    ],
    [
        # Indiana - Indianapolis
        39.5210,  # South
        39.9142,  # North
        -86.3176,  # West
        -85.9750,  # East
    ],
    [
        # Iowa - Des Moines
        41.4734,  # South
        41.6924,  # North
        -93.7676,  # West
        -93.5002,  # East
    ],
    [
        # Kansas - Wichita
        37.5721,  # South
        37.8511,  # North
        -97.5095,  # West
        -96.7523,  # East
    ],
    [
        # Kentucky - Louisville
        37.9335,  # South
        38.2975,  # North
        -85.9314,  # West
        -85.4055,  # East
    ],
    [
        # Louisiana - New Orleans
        29.7395,  # South
        30.1804,  # North
        -90.1402,  # West
        -89.8196,  # East
    ],
    [
        # Maine - Portland
        43.5520,  # South
        43.7279,  # North
        -70.2904,  # West
        -70.1870,  # East
    ],
    [
        # Maryland - Baltimore
        39.1973,  # South
        39.3710,  # North
        -76.7115,  # West
        -76.5299,  # East
    ],
    [
        # Massachusetts - Boston
        42.2279,  # South
        42.4008,  # North
        -71.1914,  # West
        -70.9234,  # East
    ],
    [
        # Michigan - Detroit
        42.2555,  # South
        42.4509,  # North
        -83.2871,  # West
        -82.9105,  # East
    ],
    [
        # Minnesota - Minneapolis
        44.6390,  # South
        45.0515,  # North
        -93.3299,  # West
        -92.9928,  # East
    ],
    [
        # Mississippi - Jackson
        32.0817,  # South
        32.3932,  # North
        -90.3140,  # West
        -89.9985,  # East
    ],
    [
        # Missouri - Kansas City
        38.8169,  # South
        39.3981,  # North
        -94.8220,  # West
        -94.4843,  # East
    ],
    [
        # Montana - Billings
        45.7272,  # South
        45.8506,  # North
        -108.6687,  # West
        -108.2147,  # East
    ],
    [
        # Nebraska - Omaha
        41.1550,  # South
        41.3723,  # North
        -96.0416,  # West
        -95.8715,  # East
    ],
    [
        # Nevada - Las Vegas
        35.9208,  # South
        36.3854,  # North
        -115.4374,  # West
        -115.0629,  # East
    ],
    [
        # New Hampshire - Manchester
        42.9279,  # South
        43.0532,  # North
        -71.5210,  # West
        -71.4022,  # East
    ],
    [
        # New Jersey - Newark
        40.6758,  # South
        40.8298,  # North
        -74.2509,  # West
        -74.0750,  # East
    ],
    [
        # New Mexico - Albuquerque
        35.0019,  # South
        35.2181,  # North
        -106.8195,  # West
        -106.4563,  # East
    ],
    [
        # New York - New York City
        40.4961,  # South
        40.9176,  # North
        -74.2591,  # West
        -73.7004,  # East
    ],
    [
        # North Carolina - Charlotte
        35.0075,  # South
        35.4049,  # North
        -81.0195,  # West
        -80.6891,  # East
    ],
    [
        # North Dakota - Fargo
        46.7358,  # South
        46.9790,  # North
        -97.1892,  # West
        -96.7898,  # East
    ],
    [
        # Ohio - Columbus
        39.7771,  # South
        40.2092,  # North
        -83.1391,  # West
        -82.8228,  # East
    ],
    [
        # Oklahoma - Oklahoma City
        35.2075,  # South
        35.6672,  # North
        -97.7551,  # West
        -97.2414,  # East
    ],
    [
        # Oregon - Portland
        45.3316,  # South
        45.6527,  # North
        -122.8366,  # West
        -122.4727,  # East
    ],
    [
        # Pennsylvania - Philadelphia
        39.8718,  # South
        40.1371,  # North
        -75.2803,  # West
        -74.9558,  # East
    ],
    [
        # Rhode Island - Providence
        41.7722,  # South
        41.8685,  # North
        -71.4641,  # West
        -71.3770,  # East
    ],
    [
        # South Carolina - Charleston
        32.7013,  # South
        32.9067,  # North
        -80.1408,  # West
        -79.8301,  # East
    ],
    [
        # South Dakota - Sioux Falls
        43.4451,  # South
        43.5859,  # North
        -96.8575,  # West
        -96.6376,  # East
    ],
    [
        # Tennessee - Memphis
        34.9816,  # South
        35.2586,  # North
        -90.2053,  # West
        -89.7025,  # East
    ],
    [
        # Texas - Houston
        29.5223,  # South
        30.1108,  # North
        -95.7972,  # West
        -95.0146,  # East
    ],
    [
        # Utah - Salt Lake City
        40.6994,  # South
        40.8090,  # North
        -112.1181,  # West
        -111.8085,  # East
    ],
    [
        # Vermont - Burlington
        44.3804,  # South
        44.5067,  # North
        -73.2981,  # West
        -73.1883,  # East
    ],
    [
        # Virginia - Virginia Beach
        36.6115,  # South
        36.9375,  # North
        -76.0485,  # West
        -75.5771,  # East
    ],
    [
        # Washington - Seattle
        47.3025,  # South
        47.7341,  # North
        -122.4597,  # West
        -122.2244,  # East
    ],
    [
        # West Virginia - Charleston
        38.2965,  # South
        38.4362,  # North
        -81.7114,  # West
        -81.5613,  # East
    ],
    [
        # Wisconsin - Milwaukee
        42.9220,  # South
        43.2411,  # North
        -88.0599,  # West
        -87.8631,  # East
    ],
    [
        # Wyoming - Cheyenne
        41.1037,  # South
        41.1885,  # North
        -104.8685,  # West
        -104.7178,  # East
    ],
    [
        # Tokyo, Japan
        35.5536,  # South
        35.8174,  # North
        139.2392,  # West
        139.9289,  # East
    ],
    [
        # New York City, USA
        40.4961,  # South
        40.9176,  # North
        -74.2591,  # West
        -73.7004,  # East
    ],
    [
        # Sao Paulo, Brazil
        -23.8162,  # South
        -23.0016,  # North
        -47.1429,  # West
        -46.3650,  # East
    ],
    [
        # Moscow, Russia
        55.4909,  # South
        56.0094,  # North
        36.8030,  # West
        37.9664,  # East
    ],
    [
        # Paris, France
        48.8156,  # South
        49.0047,  # North
        2.2241,  # West
        2.4699,  # East
    ],
    [
        # Zurich, Switzerland
        47.3203031,  # South
        47.4308514,  # North
        8.4487451,  # West
        8.5977255,  # East
    ],
    [
        # Sofia, Bulgaria
        42.6039868,  # South
        42.7831884,  # North
        23.2116106,  # West
        23.4761724,  # East
    ],
    [
        # Hyderabad, India
        17.2190976,  # South
        17.5950484,  # North
        78.1633353,  # West
        78.6518683,  # East
    ],
    [
        # Los Angeles, USA
        33.7037,  # South
        34.3373,  # North
        -118.6682,  # West
        -117.8382,  # East
    ],
    [
        # Chicago, USA
        41.6445,  # South
        42.0230,  # North
        -88.2120,  # West
        -87.5249,  # East
    ],
    [
        # Miami, USA
        25.5584,  # South
        25.9334,  # North
        -80.6158,  # West
        -80.1037,  # East
    ],
    [
        # London, United Kingdom
        51.3841,  # South
        51.6723,  # North
        -0.3517,  # West
        0.1480,  # East
    ],
    [
        # Berlin, Germany
        52.3382,  # South
        52.6755,  # North
        13.0884,  # West
        13.7600,  # East
    ],
    [
        # Rome, Italy
        41.7998,  # South
        42.0290,  # North
        12.2522,  # West
        12.8558,  # East
    ],
    [
        # Beijing, China
        39.6991,  # South
        40.2165,  # North
        115.4234,  # West
        116.9350,  # East
    ],
    [
        # Dubai, United Arab Emirates
        24.7437,  # South
        25.3548,  # North
        54.8697,  # West
        56.2045,  # East
    ],
    [
        # Singapore
        1.1496,  # South
        1.4785,  # North
        103.5901,  # West
        104.0938,  # East
    ],
    [
        # Mumbai, India
        18.8926,  # South
        19.2710,  # North
        72.7750,  # West
        72.9865,  # East
    ],
    [
        # Toronto, Canada
        43.6385,  # South
        43.8554,  # North
        -79.6392,  # West
        -79.1153,  # East
    ],
    [
        # Cape Town, South Africa
        -34.3576,  # South
        -33.7656,  # North
        18.3641,  # West
        18.8553,  # East
    ],
    [
        # Vancouver, Canada
        49.1601,  # South
        49.3148,  # North
        -123.3175,  # West
        -122.8312,  # East
    ],
    [
        # Cairo, Egypt
        29.9605,  # South
        30.1534,  # North
        31.1668,  # West
        31.3337,  # East
    ],
    [
        # Bangkok, Thailand
        13.5204,  # South
        13.9984,  # North
        100.3275,  # West
        100.9784,  # East
    ],
    [
        # Seoul, South Korea
        37.4269,  # South
        37.7017,  # North
        126.7642,  # West
        127.1839,  # East
    ],
    [
        # Sydney, Australia
        -34.1693,  # South
        -33.5781,  # North
        150.5209,  # West
        151.3430,  # East
    ],
    [
        # Mexico City, Mexico
        19.1315,  # South
        19.6659,  # North
        -99.3271,  # West
        -98.9137,  # East
    ],
    [
        # Istanbul, Turkey
        40.8027,  # South
        41.2873,  # North
        28.6325,  # West
        29.3783,  # East
    ],
    [
        # Rio de Janeiro, Brazil
        -23.0815,  # South
        -22.7469,  # North
        -43.7967,  # West
        -43.1040,  # East
    ],
    [
        # Lagos, Nigeria
        6.2641,  # South
        6.7024,  # North
        3.0988,  # West
        3.5359,  # East
    ]
]

IMAGE_RESOLUTION = 400
CROP_PIXELS = 20
MIN_ROAD_PERCENTAGE = 10.
DATA_FOLDER = "/home/ivan/PycharmProjects/ETH/road-segmentation-ethz-cil-2023/data/collected"

ZOOM_LEVEL = 18
ITEARTION_STEP = .035


def get_satelite_image(location, folder, id):
    global key
    center_x, center_y = location
    url = "https://maps.googleapis.com/maps/api/staticmap?" + \
          f"center={center_x},{center_y}" + \
          f"&zoom={ZOOM_LEVEL}&size={IMAGE_RESOLUTION + CROP_PIXELS}x{IMAGE_RESOLUTION + CROP_PIXELS}" + \
          f"&format=PNG&maptype=satellite&key={key}"
    response = requests.get(url)

    if response.status_code != 200:
        if key == KEYS[3]:
            print(response.text)
            print(response.status_code)
            exit()
        print("Keys changed")
        get_satelite_image(location, folder, id)
        key = KEYS[1]

    image = Image.open(BytesIO(response.content))
    # Get image size
    width, height = image.size

    # Crop image to remove bottom 20 pixels
    cropped_image = image.crop((0, 0, width - CROP_PIXELS, height - CROP_PIXELS))
    print(f"{DATA_FOLDER}/{folder}/{id}.png")
    cropped_image.convert("RGB").save(f"{DATA_FOLDER}/{folder}/{id}.png")


def get_street_labels(location, folder, id):
    global key
    center_x, center_y = location
    url = "https://maps.googleapis.com/maps/api/staticmap?" + \
          f"center={center_x},{center_y}&key={key}&zoom={ZOOM_LEVEL}&" + \
          f"size={IMAGE_RESOLUTION + CROP_PIXELS}x{IMAGE_RESOLUTION + CROP_PIXELS}&maptype=roadmap&format=PNG&" + \
          "style=feature:all|element:labels|visibility:off&" + \
          "style=feature:administrative|visibility:off&" + \
          "style=feature:landscape|visibility:off&" + \
          "style=feature:poi|visibility:off&" + \
          "style=feature:water|visibility:off&" + \
          "style=feature:transit|visibility:off&" + \
          "style=feature:road|element:geometry|color:0xffffff"

    response = requests.get(url)
    if response.status_code != 200:
        if key == KEYS[3]:
            print(response.text)
            print(response.status_code)
            exit()
        print("Keys changed")
        key = KEYS[3]
        get_street_labels(location, folder, id)
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
    location_folder_id = 91
    for location in locations[91:]:
        location_folder = f"{location_folder_id}_ZOOM_{ZOOM_LEVEL}"
        file_id = 1

        # By test and trial I got these number as full move in each direction without intersections between images
        # For zoom 16 - .035 
        # For zoom 15 - 0.07
        step_x = .035 if ZOOM_LEVEL == 16 else .015  # TODO: Make configs
        step_y = .035 if ZOOM_LEVEL == 16 else .015  # TODO: Make configs
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

if __name__ == '__main__':
    pass
