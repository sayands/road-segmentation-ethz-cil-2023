import requests

key = "AIzaSyBRr3cZlbyDizvYzRUJvVirv2-B5PMyj_0"

def get_satelite_image(location):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center=42.6953478,23.288339&zoom=15&size=1024x1024&format=JPEG&maptype=satellite&key={key}"

    response = requests.get(url)

    with open("gigatest.jpg", "wb") as f:
        f.write(response.content)

    
def get_street_label(location):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center=42.6953478,23.288339&&zoom=15&size=1024x1024&maptype=roadmap&format=JPEG&style=feature:all%7Celement:labels%7Cvisibility:off&style=feature:landscape%7Celement:geometry%7Cvisibility:off&style=feature:transit%7Celement:geometry%7Cvisibility:off&style=feature:administrative%7Celement:geometry.stroke%7Ccolor:0xffffff&style=feature:administrative%7Celement:geometry.fill%7Ccolor:0xffffff&style=feature:road%7Celement:geometry.stroke%7Ccolor:0x000000&key={key}"

    response = requests.get(url)
    with open("gigatest_road.jpg", "wb") as f:
        f.write(response.content)

get_street_label("")