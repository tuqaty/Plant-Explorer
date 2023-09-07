from django.urls import path
from . import views

urlpatterns = [
    path('page1',views.page1, name='page1'),
    path('',views.predicet, name='predicet'),
    path('t3',views.t3, name='t3'),
    path('asmaa',views.asmaa,name="asmaa"),
    # path('predImg1',views.predImg1,name="predImg1"),
    path('result',views.result,name="result"),
    path('predImg',views.predImg,name="predImg"),
    path('predImg2',views.predImg2,name="predImg2"),


]