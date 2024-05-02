import numpy as np
import cv2
import face_recognition
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from aiohttp import ClientSession

app = FastAPI()

class CompareRequest(BaseModel):
    url1: str
    url2: str


async def read_image_from_url(url: str):
    """
    Asynchronously fetches an image from a URL and converts it to a format suitable for processing.
    """
    async with ClientSession() as session:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                image_bytes = await response.read()
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR) 
                if image is not None:
                    return image
                else:
                    raise ValueError("Could not decode the image from the provided URL.")
        except Exception as e:
            print(f"Error reading image from URL {url}: {str(e)}")
            return None


def resize_image(image, width=100, height=100):
    """
    Resizes the image to the specified width and height.
    """
    return cv2.resize(image, (width, height))



def load_image_and_check_mask(image, image_number):
    """
    Checks for masks and returns face encodings if no mask is present.
    """
    try:
        face_locations = face_recognition.face_locations(image)
        face_landmarks_list = face_recognition.face_landmarks(image, face_locations)
        if len(face_landmarks_list) < 1:
            return None, True, image_number 
        for face_landmarks in face_landmarks_list:
            if "nose_tip" not in face_landmarks or "top_lip" not in face_landmarks:
                return None, True, image_number
        face_encodings = face_recognition.face_encodings(image, face_locations)
        return face_encodings[0] if face_encodings else None, False, None
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, False, None



def compare_faces(encoding_1, encoding_2, threshold=0.6):
    """
    Compares two face encodings and returns a match result and percentage.
    The threshold for considering a match can be adjusted to be more or less strict.
    """
    if encoding_1 is None or encoding_2 is None:
        return {"error": "One or both images do not contain a face or failed to load correctly."}
    
    distance = face_recognition.face_distance([encoding_1], encoding_2)[0]
    match = bool(distance < threshold)
    accuracy = 100 - round(distance * 100)
    
    return {"status": 1, "data":{"match": match, "accuracy": accuracy}, "message": "Success"}



@app.post("/compare-faces/")
async def compare_faces_api(request: CompareRequest):
    """
    API endpoint to compare two face images from URLs.
    """
    image_1 = await read_image_from_url(request.url1)
    image_2 = await read_image_from_url(request.url2)
    if image_1 is None or image_2 is None:
        return {"status": 0, "message": "Failed to process one or both images"}
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, message="Failed to process one or both images.")

    image_1 = resize_image(image_1)
    image_2 = resize_image(image_2)

    face_1, mask_1, image_1_number = load_image_and_check_mask(image_1, 1)
    face_2, mask_2, image_2_number = load_image_and_check_mask(image_2, 2)
    if mask_1 or mask_2:
        if mask_1:
            return {"status": 0, "message": "Please ensure that image 1 does not have a mask obscuring the face"}
            raise HTTPExceptstaion(status_code=status.HTTP_400_BAD_REQUEST, message="Please ensure that image 1 does not have a mask obscuring the face.")
        else:
            return {"status": 0, "message": "Please ensure that image 2 does not have a mask obscuring the face"}
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, message="Please ensure that image 2 does not have a mask obscuring the face.")
    THRESHOLD_VALUE = 0.5
    response = compare_faces(face_1, face_2, threshold=THRESHOLD_VALUE)
    return response



@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """
    Handles any exceptions that occur during API operations.
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "An internal server error occurred."},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8555)
