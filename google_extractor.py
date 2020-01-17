from google_images_download import google_images_download
response = google_images_download.googleimagesdownload()
lists_of_images = ["JR 103 series","JR 205 series","JR 203 Series"]

for w in lists_of_images:
    arguments = {"keywords": w, "limit": 500,"print_urls": True, "chromedriver":r"C:\chromedriver.exe"}  # creating list of arguments
    paths = response.download(arguments)  # passing the arguments to the function
    print(paths)  # printing absolute paths of the downloaded images