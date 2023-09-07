from distutils import bcppcompiler
from distutils.command.upload import upload
from multiprocessing import context
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import urls
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import torchvision.transforms.functional as TF
import torch
from PIL import Image
from applic import load_model 
from django.core.files.storage import FileSystemStorage
import json

# Create your views here.

######## pages ##########

def asmaa(request):
    return render(request,'asmaa.html');

def predicet(request):
    return render(request,'predicet.html');

def t3(request):
    return render(request,'t3.html');

def page1(request):
    return render(request,"page1.html")
###############################********functions *********##################################
############## care plant ###########
def result(request):
    if request.method == 'GET':
        context={}
        var1 = int(request.GET['p1'])
        df=pd.read_csv('media/tuqa.csv',encoding='utf-8')
        info=df.iloc[var1-1][1]
        context["gg"]=info
        s1 = df.iloc[var1-1][2]
        s2 = df.iloc[var1-1][3]
        s3 = df.iloc[var1-1][4]
        s4 = df.iloc[var1-1][5]
        print("hello")
        name = str(var1)
        path_img = "media/"+name+".jpg"
        context["url"] = path_img
        context["s1"]=s1
        context["s2"]=s2
        context["s3"]=s3
        context["s4"]=s4
    return render(request,'t3.html',context);

############# backend & test image #############
model = load_model.load_checkpoint('C:/Users/96399/Desktop/Django_project/graduation/models/checkpoint.pt')
normalize_mean = np.array([0.485, 0.456, 0.406])
normalize_std = np.array([0.229, 0.224, 0.225])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    image = TF.resize(image, 256)
    upper_pixel = (image.height - 224) // 2
    left_pixel = (image.width - 224) // 2
    image = TF.crop(image, upper_pixel, left_pixel, 224, 224)
    image = TF.to_tensor(image)
    image = TF.normalize(image, normalize_mean, normalize_std)
    return image
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image = process_image(image)
    
    with torch.no_grad():
        model.eval()
        image = image.view(1,3,224,224)
        image = image.to(device)
        predictions = model.forward(image)
        predictions = torch.exp(predictions)
        top_ps, top_class = predictions.topk(topk, dim=1)
    return top_ps, top_class

def predImg(request):
    if request.method == 'POST': 
        context = {} 
        uploaded_file= request.FILES['img'] 
        fs = FileSystemStorage() 
        name = fs.save(uploaded_file.name, uploaded_file) 
        context["url"] = fs.url(name)
        print(context["url"]) 
        testimage = '.'+context["url"] 
        probs,classes= predict(testimage,model)
        with open('media/new.json', 'r') as f:
            cat_to_name = json.load(f)
        test=pd.read_csv('media/data.csv',encoding='utf-8')
        a=classes[0,0].tolist()
        b=classes[0,1].tolist()
        c=classes[0,2].tolist()
        d=classes[0,3].tolist()
        e=classes[0,4].tolist()
        a1=cat_to_name[str(a)]
        b1=cat_to_name[str(b)]
        c1=cat_to_name[str(c)]
        d1=cat_to_name[str(d)]
        e1=cat_to_name[str(e)]
        x=classes[0,0].tolist()
        y=classes[0,1].tolist()
        z=classes[0,2].tolist()
        w=classes[0,3].tolist()
        j=classes[0,4].tolist()
        context["classes"]=a1+str(y)+"%"
        context["s1"]=b1+"\n\n"+str(y)+"%"
        context["s2"]=c1+"\n\n"+str(z)+"%"
        context["s3"]=d1+"\n\n"+str(w)+"%"
        context["s4"]=e1+"\n\n"+str(j)+"%"
        context["step1"]=test.iloc[a-1][0]
        context["step2"]=test.iloc[a-1][1]
        context["step3"]=test.iloc[a-1][2]
        context["step4"]=test.iloc[a-1][4]
        context["step5"]=test.iloc[a-1][3]
        path_img1 = "media/semi/"+str(b)+".jpg"
        context["u1"]=path_img1
        path_img2 = "media/semi/"+str(c)+".jpg"
        context["u2"]=path_img2
        path_img3 = "media/semi/"+str(d)+".jpg"
        context["u3"]=path_img3
        path_img4 ="media/semi/"+str(e)+".jpg"
        context["u4"]=path_img4
        print(a,b,c,d,e)
        return render(request,"page1.html",context)
    else:
        return render(request,"page1.html")

def predImg2(request):
    if request.method == 'GET': 
        context = {} 
        val=request.GET['p1']
        with open('media/new.json', 'r') as f:
            cat_to_name = json.load(f)
        for key, value in cat_to_name.items():
            if val == value:
                k=key
        pathimg="media/semi/" + k +".jpg"
        context["url"] = pathimg
        print(context["url"]) 
        testimage = pathimg
        probs,classes= predict(testimage,model)
        test=pd.read_csv('media/data.csv',encoding='utf-8')
        b=classes[0,1].tolist()
        c=classes[0,2].tolist()
        d=classes[0,3].tolist()
        e=classes[0,4].tolist()
        a1=cat_to_name[k]
        b1=cat_to_name[str(b)]
        c1=cat_to_name[str(c)]
        d1=cat_to_name[str(d)]
        e1=cat_to_name[str(e)]
        x=classes[0,0].tolist()
        y=classes[0,1].tolist()
        z=classes[0,2].tolist()
        w=classes[0,3].tolist()
        j=classes[0,4].tolist()
        context["classes"]=a1
        context["s1"]=b1+"\n\n"+str(y)+"%"
        context["s2"]=c1+"\n\n"+str(z)+"%"
        context["s3"]=d1+"\n\n"+str(w)+"%"
        context["s4"]=e1+"\n\n"+str(j)+"%"
        a=int(k)
        context["step1"]=test.iloc[a-1][0]
        context["step2"]=test.iloc[a-1][1]
        context["step3"]=test.iloc[a-1][2]
        context["step4"]=test.iloc[a-1][4]
        context["step5"]=test.iloc[a-1][3]
        path_img1 = "media/semi/"+str(b)+".jpg"
        context["u1"]=path_img1
        path_img2 = "media/semi/"+str(c)+".jpg"
        context["u2"]=path_img2
        path_img3 = "media/semi/"+str(d)+".jpg"
        context["u3"]=path_img3
        path_img4 ="media/semi/"+str(e)+".jpg"
        context["u4"]=path_img4
        print(a,b,c,d,e)
        return render(request,"page1.html",context)
    else:
        return render(request,"page1.html")

# def home(request):
#     return HttpResponse("Hello, Django!")

# def predImg1(request):
#     if request.method == 'POST':
        
#         context = {}
#         uploaded_file= request.FILES['img']
#         fs = FileSystemStorage()
#         name = fs.save(uploaded_file.name, uploaded_file)
#         context["url"] = fs.url(name)
#         print(context["url"])
#         df=pd.read_csv('media/data.csv',encoding='utf-8')
#         info=df.iloc[0][0]
#         context["gg"]=info
#     return render(request,'asmaa.html',context);

# def test(request):
#     fs = FileSystemStorage()
#     var1 = request.GET['1']
#     context["url"]=fs.url(var1)
#     return render(request,'t3.html',context);