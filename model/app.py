# In this code i am using integrated cam of my laptop to capture image and then using that image to get the description of the image using Llama API.
# To run this code make sure to get the client_id from imgur and API key from Hugging Face. And also change the camera settings as mine is integrated one

import cv2
import requests
import json
from huggingface_hub import InferenceClient

# Function to upload an image to Imgur and return the URL
def upload_image_to_imgur(image_path, client_id):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    headers = {
        'Authorization': f'Client-ID {client_id}',
    }

    payload = {
        'image': image_data,
        'type': 'file',
    }

    response = requests.post('https://api.imgur.com/3/upload', headers=headers, files=payload)

    if response.status_code == 200:
        image_url = response.json()['data']['link']
        return image_url
    else:
        print(f"Failed to upload image: {response.status_code}, {response.text}")
        return None

# Function to capture an image from the camera and save it locally
def capture_image_from_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return None

    print(f"Using camera {camera_index}. Press 's' to capture an image.")
    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 's' to capture
            image_path = 'captured_image.jpg'
            cv2.imwrite(image_path, frame)
            print(f"Image saved as {image_path}")
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return image_path

# Function to call Llama API and get description for the image URL
def get_image_description_from_llama(image_url):
    client = InferenceClient(api_key="")  # Obtain an API key by registering on Hugging Face

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Identify the given object. Provide the item name followed by a list of electrical components used in the object. List only the component names (e.g., Microcontroller, Button Switch, Optical Sensor) without any descriptions or additional details. Do not describe how the components work. Explain in technical terms suitable for a PCB Designer or Electronics/Hardware Engineer."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ]

    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct", 
            messages=messages, 
            max_tokens=500
        )
        return completion.choices[0].message['content']
    except Exception as e:
        print(f"Error calling Llama API: {e}")
        return None

# Function to save the description in JSON format
def save_description_to_json(description, output_file='description.json'):
    # Parse the description and convert it to the desired JSON structure
    lines = description.split('\n')
    item_name = lines[0].replace("Item Name: ", "").strip()
    item_components = [line.strip() for line in lines[2:] if line.strip()]

    # Create a dictionary with item name and components
    data = {
        "Item Name": item_name,
        "Item Components": item_components
    }

    # Save to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Description saved to {output_file}")

# Example usage
client_id = ""  # Obtain a client ID by registering on Imgur

# Capture the image from an external camera
image_path = capture_image_from_camera(camera_index=1)  # Change to camera_index=1 or another number if needed

if image_path:
    # Upload the captured image to Imgur
    image_url = upload_image_to_imgur(image_path, client_id)
    if image_url:
        print("Image URL:", image_url)

        # Get the image description from Llama
        description = get_image_description_from_llama(image_url)
        if description:
            print("Image Description:", description)

            # Save the description to a JSON file
            save_description_to_json(description)
        else:
            print("No description received from Llama.")
    else:
        print("Image upload failed.")
