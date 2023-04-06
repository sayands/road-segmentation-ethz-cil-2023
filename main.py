import requests

key = "AIzaSyBRr3cZlbyDizvYzRUJvVirv2-B5PMyj_0"

city_coordinates = [[
  # Tokyo, Japan
  (35.6217, 139.4447),  # westernmost
  (35.6845, 139.9247),  # easternmost
  (35.7769, 139.7690),  # northernmost
  (35.5431, 139.7315),  # southernmost
],[
  # New York City, USA
  (40.4984, -74.2589),  # westernmost
  (40.9176, -73.7001),  # easternmost
  (40.8790, -73.9614),  # northernmost
  (40.4774, -74.2591),  # southernmost
],[
  # Sao Paulo, Brazil
  (-23.4924, -46.3931),  # westernmost
  (-23.6808, -46.8269),  # easternmost
  (-23.7919, -46.8111),  # northernmost
  (-23.9717, -46.7347),  # southernmost
],[
  # Moscow, Russia
  (55.5036, 37.3126),  # westernmost
  (55.9723, 37.9823),  # easternmost
  (56.0097, 37.7202),  # northernmost
  (55.4896, 37.2265),  # southernmost
],[
  # Istanbul, Turkey
  (41.0092, 28.9022),  # westernmost
  (41.1239, 29.3973),  # easternmost
  (41.2034, 28.9584),  # northernmost
  (40.8022, 28.8668),  # southernmost
]]


def get_satelite_image(location, id):
    center_x, center_y = location
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={center_x},{center_y}&zoom=15&size=1024x1024&format=JPEG&maptype=satellite&key={key}"

    response = requests.get(url)

    with open(f"data/{id}.jpg", "wb") as f:
        f.write(response.content)

    
def get_street_label(location, id):
    center_x, center_y = location
    url = f"https://maps.googleapis.com/maps/api/staticmap?" +\
      "center={center_x},{center_y}&&"+\
        "zoom=15&size=1024x1024&maptype=roadmap&format=JPEG&style=feature:all%7Celement:labels%7Cvisibility:off&"+\
        "style=feature:landscape%7Celement:geometry%7Cvisibility:off&"+\
        "style=feature:transit%7Celement:geometry%7Cvisibility:off&"+\
        "style=feature:administrative%7Celement:geometry.stroke%7Ccolor:0xffffff&"+\
        "style=feature:administrative%7Celement:geometry.fill%7Ccolor:0xffffff&"+\
        "style=feature:road.highway%7Celement:geometry.stroke%7Ccolor:0xffffff&"+\
        "style=feature:transit.line|element:geometry.stroke|color:0xffffff&"+\
        "style=feature:road%7Celement:geometry.stroke%7Ccolor:0x000000"+\
        "&key={key}"
    print(url)
    exit()
    response = requests.get(url)
    with open(f"data/{id}_label.jpg", "wb") as f:
        f.write(response.content)

def get_data(locations, scan_x, scan_y):
    file_id = 1
    for location in locations:
        print(location)
        distance_x = abs(location[1][0] - location[0][0])
        step_x = distance_x / scan_x

        distance_y = abs(location[3][1] - location[2][1])
        step_y = distance_y / scan_y

        end_x, end_y = location[1][0], location[2][1]
        start_x, start_y = location[0][0], location[3][1]

        print(start_x, end_x, start_y, end_y)
        while start_x <= end_x:
            while start_y <= end_y:
                print(start_x, start_y)
                get_satelite_image((start_x, start_y),file_id)
                get_street_label((start_x, start_y), file_id)
                start_y += step_y
                file_id += 1
            start_x += step_x
        
get_data(city_coordinates, 2, 2)