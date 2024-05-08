import numpy as np
import cv2
import dlib
from math import sqrt
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from aiohttp import ClientSession
from typing import Optional

from cache import cache


app = FastAPI()

class CompareRequest(BaseModel):
    """
    Pydantic model for comparing faces.
    """
    url1: str
    url2: str

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("face/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("face/dlib_face_recognition_resnet_model_v1.dat")


async def read_image_from_url(url: str) -> Optional[np.ndarray]:
    """
    Read image from URL asynchronously.

    Args:
        url (str): URL of the image.

    Returns:
        Optional[np.ndarray]: Decoded image as a NumPy array.
    """
    cached_image = await cache.get(url)
    if cached_image is not None:
        return cached_image

    async with ClientSession() as session:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                image_bytes = await response.read()
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    await cache.set(url, image)
                    return image
                else:
                    raise ValueError("Could not decode the image from the provided URL.")
        except Exception as e:
            print(f"Error reading image from URL {url}: {str(e)}")
            return None


def resize_image(image: np.ndarray, width: int = 800, height: int = 800) -> np.ndarray:
    """
    Resize image to specified dimensions.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        width (int): Width of the resized image.
        height (int): Height of the resized image.

    Returns:
        np.ndarray: Resized image as a NumPy array.
    """
    return cv2.resize(image, (width, height))


def get_face_descriptor(image: np.ndarray):
    """
    Get face descriptor from an image.

    Args:
        image (np.ndarray): Input image as a NumPy array.

    Returns:
        Tuple: Face descriptor, boolean indicating if no face is detected, and error code.
    """
    try:
        faces = detector(image, 1)
        if len(faces) > 0:
            shape = predictor(image, faces[0])  

            nose_landmarks = shape.parts()[30:36]
            mouth_landmarks = shape.parts()[48:68]

            if any(l.x == 0 and l.y == 0 for l in nose_landmarks) or any(l.x == 0 and l.y == 0 for l in mouth_landmarks):
                return None, False, 2 
            else:
                face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
                return np.array(face_descriptor), False, None
        else:
            return None, True, 1 
    except Exception as e:
        print(f"Error processing image for face descriptor: {str(e)}")
        return None, False, None


def compute_euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two vectors.

    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Euclidean distance between the two vectors.
    """
    return sqrt(sum((v1 - v2) ** 2 for v1, v2 in zip(vec1, vec2)))


# def compare_faces(descriptor1: np.ndarray, descriptor2: np.ndarray, threshold: float = 0.6) -> dict:
#     """
#     Compare face descriptors and determine if they match.

#     Args:
#         descriptor1 (np.ndarray): Face descriptor of the first image.
#         descriptor2 (np.ndarray): Face descriptor of the second image.
#         threshold (float): Threshold for matching faces.

#     Returns:
#         dict: Dictionary containing match status and accuracy.
#     """
#     if descriptor1 is None or descriptor2 is None:
#         return {"error": "One or both images do not contain a face or failed to load correctly."}
    
#     euclidean_distance = compute_euclidean_distance(descriptor1, descriptor2)
#     match = bool(euclidean_distance < threshold)
#     accuracy = 100 - round(euclidean_distance * 100, 2)

#     if match == True:   
#         return {"status": 1, "data":{"match": match, "accuracy": accuracy}, "message": "Success"}
#     else:
#         return {"status": 1, "data":{"match": match, "accuracy": accuracy}, "message": "Error"}


def compare_faces(descriptor1: np.ndarray, descriptor2: np.ndarray, threshold: float = 0.6) -> dict:
    """
    Compare face descriptors and determine if they match.

    Args:
        descriptor1 (np.ndarray): Face descriptor of the first image.
        descriptor2 (np.ndarray): Face descriptor of the second image.
        threshold (float): Threshold for matching faces.

    Returns:
        dict: Dictionary containing match status and accuracy.
    """
    if descriptor1 is None or descriptor2 is None:
        return {"error": "One or both images do not contain a face or failed to load correctly."}
    
    euclidean_distance = compute_euclidean_distance(descriptor1, descriptor2)
    accuracy = 100 - round(euclidean_distance * 100, 2)
    
    if accuracy < 51:
        return {"status": 0, "data": {"match": False, "accuracy": accuracy}, "message": "Error"}
    else:
        match = bool(euclidean_distance < threshold)
        return {"status": 1, "data": {"match": match, "accuracy": accuracy}, "message": "Success"}
    

@app.post("/compare-faces/")
async def compare_faces_api(request: CompareRequest):
    """
    API endpoint for comparing faces from two images.

    Args:
        request (CompareRequest): Request containing URLs of two images.

    Returns:
        JSONResponse: JSON response containing match status and accuracy.
    """
    image1 = await read_image_from_url(request.url1)
    image2 = await read_image_from_url(request.url2)
    if image1 is None or image2 is None:
        return {"status": 0, "message": "Failed to process one or both images"}

    image1 = resize_image(image1)
    image2 = resize_image(image2)

    descriptor1, no_face1, _ = get_face_descriptor(image1)
    descriptor2, no_face2, _ = get_face_descriptor(image2)
    if no_face1 or no_face2:
        
        return {"status": 0, "message": "Please ensure that image 1 does not have a mask obscuring the face." if no_face1 else "Please ensure that image 2 does not have a mask obscuring the face"}

    response = compare_faces(descriptor1, descriptor2)
    return response


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """
    Exception handler for handling internal server errors.

    Args:
        request: The request that caused the exception.
        exc: The exception that was raised.

    Returns:
        JSONResponse: JSON response containing an error message.
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "An internal server error occurred."},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8555)
