import requests

key = "AIzaSyBRr3cZlbyDizvYzRUJvVirv2-B5PMyj_0"

url = f"https://maps.googleapis.com/maps/api/staticmap?center=42.6953478,23.288339&zoom=15&size=1024x1024&format=JPEG&maptype=satellite&key={key}"

response = requests.get(url)

with open("gigatest.jpg", "wb") as f:
    f.write(response.content)