import requests

url = 'http://localhost:5000/results'

r = requests.post(url,json={'barrio_cat':5, 'ambientes':200})

print(r.json())