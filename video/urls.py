from django.urls import path
from . import views

urlpatterns = [

    path('video', views.video),
    path('getUrl', views.getUrl),

    path('video2', views.video2),
    path('getUrl2', views.getUrl2),

    path('emotionList', views.emotionList),
    path('emotionDelete', views.emotionDelete),
    path('emotionDetailByID', views.emotionDetailByID),

    path('fallList', views.fallList),
    path('fallDelete', views.fallDelete),

    path('unknowList', views.unknowList),
    path('unknowDelete', views.unknowDelete),

    path('video3', views.video3),
    path('getUrl3', views.getUrl3),

    path('getXY', views.getXY),
    path('video4', views.video4),
    path('getUrl4', views.getUrl4),

    path('intrusionList', views.intrusionList),
    path('intrusionDelete', views.intrusionDelete),

    path('video5', views.video5),
    path('getUrl5', views.getUrl5),

    path('reactionList', views.reactionList),
    path('reactionDelete', views.reactionDelete),
    path('reactionDetailByID', views.reactionDetailByID),


]