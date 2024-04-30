from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import requests
import numpy as np
import cv2
import face_recognition

app = FastAPI()

def read_image_from_url(url: str):
    """
    Fetches an image from a URL and converts it to a format suitable for processing.
    Args:
        url (str): URL of the image.
    Returns:
        numpy.ndarray: The image in an array format suitable for face recognition, or None if conversion fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            return image
        else:
            raise ValueError("Could not decode the image from the provided URL.")
    except Exception as e:
        print(f"Error reading image from URL {url}: {str(e)}")
        return None

def load_image_and_find_face_encoding(image):
    """
    Processes an image to find face encodings.
    Args:
        image (numpy.ndarray): The image array to process.
    Returns:
        list: The first found face encoding, or None if no face is detected.
    """
    try:
        face_encodings = face_recognition.face_encodings(image)
        return face_encodings[0] if face_encodings else None
    except Exception as e:
        print(f"Error finding face encodings: {str(e)}")
        return None

def compare_faces(encoding_1, encoding_2):
    """
    Compares two face encodings to determine if they are of the same person.
    Args:
        encoding_1 (list): The first face encoding.
        encoding_2 (list): The second face encoding.
    Returns:
        dict: A dictionary containing the match result and accuracy.
    """
    if encoding_1 is None or encoding_2 is None:
        return {"error": "One or both images do not contain a face or failed to load correctly."}
    match_results = face_recognition.compare_faces([encoding_1], encoding_2)
    if match_results[0]:
        distance = face_recognition.face_distance([encoding_1], encoding_2)[0]
        accuracy = 100 - round(distance * 100)
        return {"match": True, "accuracy": accuracy}
    else:
        return {"match": False, "accuracy": 0}

@app.post("/compare-faces/")
async def compare_faces_api(url1: str, url2: str):
    """
    API endpoint to compare two face images from URLs to check if they are of the same person.
    """
    image_1 = read_image_from_url(url1)
    image_2 = read_image_from_url(url2)
    if image_1 is None or image_2 is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to process one or both images.")
    
    face_1 = load_image_and_find_face_encoding(image_1)
    face_2 = load_image_and_find_face_encoding(image_2)
    response = compare_faces(face_1, face_2)
    return response

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """
    Generic exception handler for the FastAPI app.
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "An internal server error occurred."},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8555)
