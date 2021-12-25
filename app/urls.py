from django.conf.urls import url

from . import views
from django.urls import path

# from .views import BasicUploadView
urlpatterns = [
    path('abjad', views.mainview.as_view(), name='mainview'),

    path('up', views.fileUploadView, name='uploadView'),
    #
    path('text/<id>', views.textView, name='textView'),

]
