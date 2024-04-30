import cv2
import face_recognition
import requests
import numpy as np
from io import BytesIO

def download_image(url):
    """
    Downloads an image from a URL and converts it into a format that cv2 can use.
    
    Args:
    url (str): URL of the image to download.

    Returns:
    numpy.ndarray: An image array that can be used with cv2.
    """
    response = requests.get(url)
    image_bytes = BytesIO(response.content)
    image = np.asarray(bytearray(image_bytes.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def load_image_and_find_face_encoding(image):
    """
    Finds the first face encoding in an image.
    
    Args:
    image (numpy.ndarray): The image in which to find the face encoding.

    Returns:
    numpy.ndarray or None: The face encoding if a face is found, otherwise None.
    """
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        return face_encodings[0]
    return None

def compare_faces(encoding_1, encoding_2):
    """
    Compares two face encodings and calculates the accuracy if they match, returning a structured response.

    Args:
    encoding_1 (numpy.ndarray): The first face encoding.
    encoding_2 (numpy.ndarray): The second face encoding.

    Returns:
    dict: A dictionary containing the match result and accuracy or error message.
    """
    if encoding_1 is None or encoding_2 is None:
        return {"error": "One or both images do not contain a face."}
    
    match_results = face_recognition.compare_faces([encoding_1], encoding_2)
    if match_results[0]:
        distance = face_recognition.face_distance([encoding_1], encoding_2)[0]
        accuracy = 100 - round(distance * 100)
        return {"match": True, "accuracy": accuracy}
    else:
        return {"match": False, "accuracy": 0}


