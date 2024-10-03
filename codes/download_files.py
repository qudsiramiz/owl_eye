import requests
from bs4 import BeautifulSoup
import os

# URL of the page containing the .tif files
url = "https://svs.gsfc.nasa.gov/vis/a000000/a005100/a005187/frames/1920x1080_16x9_30p/plain"

# Make a request to fetch the HTML content of the page
response = requests.get(url)
response.raise_for_status()  # Check for successful request

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")

# Find all links that end with .tif
tif_links = [
    link.get("href") for link in soup.find_all("a") if link.get("href").endswith(".tif")
]

# Create a directory to save the files if it doesn't exist
os.makedirs("tif_files", exist_ok=True)

# Download each .tif file
for tif_link in tif_links:
    # Complete the URL if necessary
    tif_url = url + tif_link
    print(f"Downloading {tif_url}...")

    # Request the .tif file
    tif_response = requests.get(tif_url)
    tif_response.raise_for_status()

    # Save the file
    file_name = os.path.join("tif_files", os.path.basename(tif_link))
    with open(file_name, "wb") as file:
        file.write(tif_response.content)
    print(f"Saved {file_name}")

print("All files downloaded.")
