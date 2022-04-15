import json
from django.shortcuts import render
from django.http.response import HttpResponse
from django.views import View
from .models import Doc
from django.http import JsonResponse
from django.views.generic import TemplateView
import numpy
import cv2
from PIL import Image
from As.system import System


class mainview(TemplateView):
    template_name = 'home.html'


def fileUploadView(request):
    print(request.FILES)  # check print
    print(request.method)  # check print

    if request.method == 'POST':
        myFile = request.FILES.get('file')

        Doc.objects.create(upload=myFile, Image_Name="ImageName", )

        photos_list = Doc.objects.all()

        print(Doc.objects.count())  # check print

        return HttpResponse('')

    elif request.method == 'GET':
        
        photos_list = Doc.objects.all()
        
        sys()  # Processing the Images
        for i in photos_list:
            print('photo#', i)  # check print
        context = {'photos': photos_list}
        return render(request, 'upload.html', context)

    return JsonResponse({'post': 'fales'})


def textView(request, id):
    image_obj = Doc.objects.get(id=id)

    file_Name = ''

    text = image_obj.text

    image = image_obj.upload

    fileName = toFileName(image)    # toFileName convert the image name to a valid name for the file

    filePath = f'media/text_Files/{fileName}.txt'

    fileText = open(filePath, 'w', encoding='utf-8')
    print('textFileCreated!!!!!!!!!!')
    fileText.write(f'{text}')

    print('Text:\n', text)
    context = {'text': text, 'image': image, 'fileName': fileName}
    return render(request, 'text.html', context)


def sys():
    photo_List = Doc.objects.all().filter(text='')
    print(photo_List)

    print('function start At sys/ln 99/n')
    print(type(photo_List))
    photos = []  # All objects in this list will be processed
    print(photos,
          '/n------------------------------------------------------Inatial photos------------------------------------------------------')
    if len(photo_List) == 0:
        print(
            '         Error!!! \n         -User Didnt Upload Image or the image already processed,\n         at least Upload one image with file type of .jpg, .png, or .tiff')
    else:
        print(photo_List)

        for i, obj in enumerate(photo_List):
            print(obj.text, 'obj.text/ln 106/n')

            photo = cv2.imread(str(obj.upload.file))
            print(i)
            photos.append(photo)

        print(photos,
              '/n------------------------------------------------------photos------------------------------------------------------')

        print(
            '/n------------------------------------------------------photos entered to System()------------------------------------------------------')
        print(len(photos), 'lenOfPhotos')
        text_List = System(photos)
        ListOfText = []
        for i, te in enumerate(text_List):
            ListOfText.append(text_List[i])
            print(i, '-:', text_List[i], type(text_List[i]))
            print('-:', ListOfText[i], type(ListOfText[i]))
        text_List = 'error' if len(text_List) == 0 else text_List[0]
        print(text_List)
        photos_list_ids = []
        print(photo_List.all().values('id'),'photoList.id')
        for i in range(0, len(photo_List)):
            x = photo_List.all().values('id')[i]['id']
            print(x)
            print('i', i)

            print(photos_list_ids)

            photos_list_ids.append(x)
            print(photos_list_ids[i])
        print(photos_list_ids)
        print(text_List,
              '/n------------------------------------------------------TextList------------------------------------------------------')
        print(
            '------------------------------------------------------start to add the list text to the objects text------------------------------------------------------')
        print('ListOfText: ', ListOfText, '/ntype:  ', type(ListOfText), 'sys/ln 134')
        print(len(ListOfText))
        for i, d in enumerate(text_List):
            text = []
            print(
                '------------------------------------------------------ListOftext Loop Start------------------------------------------------------')
            print(i)
            print(photos_list_ids[i], 'PhotosListIds At line148')
            x = photos_list_ids[i]
            photo_obj = photo_List.get(id=x)
            print('photo_obj: ', photo_obj, '/ntype:  ', type(photo_obj), 'sys/ln 141')
            text = text_List[i]
            photo_obj.text = text
            photo_obj.save()
            print('textList', text_List[i], type(text_List[i]))
            print('text: ', text, '/ntype:  ', type(text))
            print('ListOfText: ', ListOfText, '/ntype:  ', type(ListOfText), 'sys/ln 134')
            print('photo_obj: ', photo_obj, '/ntype:  ', type(photo_obj), 'sys/ln 146')
            print(
                '------------------------------------------------------End------------------------------------------------------')
        print(photos_list_ids, 'photoListids At Line #158')
        print(ListOfText, 'ListOftext At line #159')


def fromDictToStr(dict):
    Str = list(map(list, (ele for ele in dict.values())))
    return str(Str)



def toFileName(image):
    fullImageName = str(image.name)
    imagename = ''
    for fileName, letter in enumerate(fullImageName):

        if letter == '.':

            break
        else:
            imagename += letter

    print(f' imagename:\n', imagename)
    fileName = imagename[6:len(imagename)]
    print('Valid file name:\n', fileName)
    return fileName

