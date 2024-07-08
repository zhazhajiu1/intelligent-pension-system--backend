from django.urls import path
from . import views

urlpatterns = [
    path('login', views.login),
    path('register', views.register),
    path('logout', views.logout),
    path('userInfo', views.userInfo),
    path('userUpdateInfo', views.userUpdateInfo),
    path('employeeList', views.employeeList),
    path('employeeAdd', views.employeeAdd),
    path('employeeDelete', views.employeeDelete),
    path('employeeUpdate', views.employeeUpdate),
    path('employeeDetailByID', views.employeeDetailByID),

    path('volunteerList', views.volunteerList),
    path('volunteerAdd', views.volunteerAdd),
    path('volunteerDelete', views.volunteerDelete),
    path('volunteerUpdate', views.volunteerUpdate),
    path('volunteerDetailByID', views.volunteerDetailByID),

    path('elderlyList', views.elderlyList),
    path('elderlyAdd', views.elderlyAdd),
    path('elderlyDelete', views.elderlyDelete),
    path('elderlyUpdate', views.elderlyUpdate),
    path('elderlyDetailByID', views.elderlyDetailByID),
    path('uploadCloud', views.uploadCloud),

    path('chat', views.chat),



    # path('list', views.userlist),
    # 删除user
    # path('delete', views.deleteUser),
    # 修改用户
    # path('update', views.updateUser),
    # path('detail', views.detail),
    # path('add', views.add)

]