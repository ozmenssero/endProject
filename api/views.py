from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
# Create your views here.
from .bitirmefrangideneme import calculateLesionLength
import cv2
import numpy as np
import json

@api_view(['POST'])
def createNote(request):
    if request.method == 'POST' and request.FILES['image']:
        image=request.FILES['image']
        lesionParameters=json.loads(request.data['lesionParameters'])
    imageMatrix = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    lesionLength=calculateLesionLength(imageMatrix,lesionParameters)
    print(f"api sonuc:{lesionLength}")
    # print(f"request data??: {lesionParameters}")
    # print(f"_______________")
    # print(f"request data??: {type(lesionParameters)}")
    return Response([lesionLength,])


@api_view(['GET'])
def getNotes(request):
    return Response("notes")


