import requests

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"text": "Scientists discover that drinking water cures all diseases instantly"}
)

print(response.json())