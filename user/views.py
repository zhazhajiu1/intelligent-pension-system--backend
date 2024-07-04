from __future__ import unicode_literals

import binascii
import json
import os
from django.http import JsonResponse
from . import redis_connect
from .models import *
import oss2
import requests

# 填写您的Access Key ID和Access Key Secret
access_key_id = 'LTAI5tR85Q78cGEzRpagmsjh'
access_key_secret = 'JMIJbsWZP6sBPAkxEaQdONt52NSqBx'
# 填写您的Bucket所在地域
endpoint = 'https://oss-cn-beijing.aliyuncs.com'
# 填写Bucket名称
bucket_name = 'old-care-bucket'
# 创建Bucket对象
bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)


def getImgUrl(img):
    if bucket:
        url = bucket.sign_url('GET', img, 10 * 60)
        return url
    else:
        return 'not link'


def getRoleByToken(token):
    result = json.loads(redis_connect.get(token))
    return result.get('role')


def getIDByToken(token):
    result = json.loads(redis_connect.get(token))
    return result.get('ID')


def getUserNameByToken(token):
    result = json.loads(redis_connect.get(token))
    return result.get('UserName')


# Create your views here.
def login(request):
    post_body = request.body
    # print(post_body)
    json_param = json.loads(post_body.decode())
    # print(json_param)

    UserRole = json_param.get('UserRole')
    UserName = json_param.get('UserName')
    Password = json_param.get('Password')
    # print(UserRole)

    if UserRole == '0':
        result = Admin.objects.filter(UserName=UserName, Password=Password).first()
        # print(result)
        if result:
            token = binascii.hexlify(os.urandom(20)).decode()
            _dict = {'UserName': UserName, 'role': 'admin', 'ID': result.ID}
            redis_connect.set(token, json.dumps(_dict), 259200)
            data = {"token": token}

            return JsonResponse({'code': 20000, 'message': '登录成功', "data": data})
        else:
            return JsonResponse({'code': 20001, 'message': '用户名或密码错误'})
    elif UserRole == '1':
        result = Employee.objects.filter(UserName=UserName, Password=Password).first()
        if result and result.IsActive == '0':
            token = binascii.hexlify(os.urandom(20)).decode()
            _dict = {'UserName': UserName, 'role': 'employee', 'ID': result.ID}
            redis_connect.set(token, json.dumps(_dict), 259200)
            data = {"token": token}
            return JsonResponse({'code': 20000, 'message': '登录成功', "data": data})
        else:
            return JsonResponse({'code': 20001, 'message': '用户名或密码错误或用户未启用'})
    else:
        result = Volunteer.objects.filter(UserName=UserName, Password=Password).first()
        if result and result.IsActive == '0':
            token = binascii.hexlify(os.urandom(20)).decode()
            _dict = {'UserName': UserName, 'role': 'volunteer', 'ID': result.ID}
            redis_connect.set(token, json.dumps(_dict), 259200)
            data = {"token": token}
            return JsonResponse({'code': 20000, 'message': '登录成功', "data": data})
        else:
            return JsonResponse({'code': 20001, 'message': '用户名或密码错误或用户未启用'})


def register(request):
    post_body = request.body
    # print(post_body)
    json_param = json.loads(post_body.decode())
    # print(json_param)

    UserRole = json_param.get('UserRole')
    UserName = json_param.get('UserName')
    Password = json_param.get('Password')
    Sex = json_param.get('Sex')
    Age = json_param.get('Age')
    Phone = json_param.get('Phone')

    if UserRole == '1':
        result = Employee.objects.filter(UserName=UserName)
        if result.exists():
            return JsonResponse({'code': 20002, 'message': '用户已存在'})

        result1 = Employee.objects.filter(Phone=Phone)
        if result1.exists():
            return JsonResponse({'code': 20003, 'message': '手机号已被注册'})

        if not result.exists() and not result1.exists():
            employee = Employee(UserName=UserName,
                                Password=Password,
                                Sex=Sex,
                                Age=Age,
                                Phone=Phone)
            employee.save()

            return JsonResponse({'code': 20000, 'message': '成功注册'})
    else:
        ImgUrl = json_param.get('ImgUrl')
        result = Volunteer.objects.filter(UserName=UserName)
        if result.exists():
            return JsonResponse({'code': 20002, 'message': '用户已存在'})

        result1 = Volunteer.objects.filter(Phone=Phone)
        if result1.exists():
            return JsonResponse({'code': 20003, 'message': '手机号已被注册'})

        if not result.exists() and not result1.exists():
            volunteer = Volunteer(UserName=UserName,
                                  Password=Password,
                                  Sex=Sex,
                                  Age=Age,
                                  Phone=Phone,
                                  ImgUrl=ImgUrl)
            volunteer.save()
            return JsonResponse({'code': 20000, 'message': '成功注册'})


def logout(request):
    return JsonResponse({'code': 20000, 'message': '注销成功'})


def userInfo(request):
    print(request)
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})

    # print(token)
    # print(result)
    # print(result.get('username'))
    role = getRoleByToken(token)
    # ID = getIDByToken(token)
    UserName = getUserNameByToken(token)
    if role == 'admin':
        result = Admin.objects.filter(UserName=UserName).first()
        return JsonResponse({
            "code": 20000,
            "data": {
                'ID': result.ID,
                'UserName': result.UserName,
                'Password': result.Password,
                'Sex': result.Sex,
                'Phone': result.Phone,
                'IsActive': result.IsActive,
                'Created': result.Created,
                'Updated': result.Updated
            }
        })
    elif role == 'employee':
        result = Employee.objects.filter(UserName=UserName).first()
        return JsonResponse({
            "code": 20000,
            "data": {
                'ID': result.ID,
                'UserName': result.UserName,
                'Password': result.Password,
                'Sex': result.Sex,
                'Age': result.Age,
                'Phone': result.Phone,
                'IsActive': result.IsActive,
                'Created': result.Created,
                'Updated': result.Updated
            }
        })
    elif role == 'volunteer':
        result = Volunteer.objects.filter(UserName=UserName).first()
        url = getImgUrl(result.ImgUrl)
        return JsonResponse({
            "code": 20000,
            "data": {
                'ID': result.ID,
                'UserName': result.UserName,
                'Password': result.Password,
                'Sex': result.Sex,
                'Age': result.Age,
                'Phone': result.Phone,
                'ImgUrl': result.ImgUrl,
                'Url': url,
                'IsActive': result.IsActive,
                'Created': result.Created,
                'Updated': result.Updated
            }
        })


def userUpdateInfo(request):
    post_body = request.body
    json_param = json.loads(post_body.decode())
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    # result = json.loads(redis_connect.get(token))
    # print(result)
    # print(result.get('username'))
    # role = result.get('role')
    # UserName = result.get('UserName')
    # ID = result.get('ID')
    role = getRoleByToken(token)
    ID = getIDByToken(token)
    if role == 'admin':
        UserName = json_param.get('UserName')
        Password = json_param.get('Password')
        Sex = json_param.get('Sex')
        Phone = json_param.get('Phone')
        IsActive = json_param.get('IsActive')

        results = Admin.objects.filter(UserName=UserName)
        if results.exists():
            for admin in results:
                if admin.ID != ID:
                    return JsonResponse({'code': 20003, 'message': '用户名已被注册'})
                else:
                    Admin.objects.filter(ID=ID).update(UserName=UserName)
                    Admin.objects.filter(ID=ID).update(Password=Password)
                    Admin.objects.filter(ID=ID).update(Sex=Sex)
                    Admin.objects.filter(ID=ID).update(Phone=Phone)
                    Admin.objects.filter(ID=ID).update(IsActive=IsActive)
                    return JsonResponse({'code': 20000, 'message': '成功修改'})

        results1 = Admin.objects.filter(Phone=Phone)
        if results1.exists():
            for admin in results1:
                if admin.ID != ID:
                    return JsonResponse({'code': 20003, 'message': '手机号已被注册'})
                else:

                    Admin.objects.filter(ID=ID).update(UserName=UserName)
                    Admin.objects.filter(ID=ID).update(Password=Password)
                    Admin.objects.filter(ID=ID).update(Sex=Sex)
                    Admin.objects.filter(ID=ID).update(Phone=Phone)
                    Admin.objects.filter(ID=ID).update(IsActive=IsActive)
                    return JsonResponse({'code': 20000, 'message': '成功修改'})

        if not (results1.exists() or results.exists()):
            Admin.objects.filter(ID=ID).update(UserName=UserName)
            Admin.objects.filter(ID=ID).update(Password=Password)
            Admin.objects.filter(ID=ID).update(Sex=Sex)
            Admin.objects.filter(ID=ID).update(Phone=Phone)
            Admin.objects.filter(ID=ID).update(IsActive=IsActive)
            return JsonResponse({'code': 20000, 'message': '成功修改'})

    elif role == 'employee':
        UserName = json_param.get('UserName')
        Password = json_param.get('Password')
        Sex = json_param.get('Sex')
        Age = json_param.get('Age')
        Phone = json_param.get('Phone')
        IsActive = json_param.get('IsActive')

        results = Employee.objects.filter(UserName=UserName)
        if results.exists():
            for employee in results:
                if employee.ID != ID:
                    return JsonResponse({'code': 20003, 'message': '用户名已被注册'})
                else:
                    Employee.objects.filter(ID=ID).update(UserName=UserName)
                    Employee.objects.filter(ID=ID).update(Password=Password)
                    Employee.objects.filter(ID=ID).update(Sex=Sex)
                    Employee.objects.filter(ID=ID).update(Age=Age)
                    Employee.objects.filter(ID=ID).update(Phone=Phone)
                    Employee.objects.filter(ID=ID).update(IsActive=IsActive)
                    return JsonResponse({'code': 20000, 'message': '成功修改'})

        results1 = Employee.objects.filter(Phone=Phone)
        if results1.exists():
            for employee in results1:
                if employee.ID != ID:
                    return JsonResponse({'code': 20003, 'message': '手机号已被注册'})
                else:

                    Employee.objects.filter(ID=ID).update(UserName=UserName)
                    Employee.objects.filter(ID=ID).update(Password=Password)
                    Employee.objects.filter(ID=ID).update(Sex=Sex)
                    Employee.objects.filter(ID=ID).update(Age=Age)
                    Employee.objects.filter(ID=ID).update(Phone=Phone)
                    Employee.objects.filter(ID=ID).update(IsActive=IsActive)
                    return JsonResponse({'code': 20000, 'message': '成功修改'})

        if not (results1.exists() or results.exists()):
            Employee.objects.filter(ID=ID).update(UserName=UserName)
            Employee.objects.filter(ID=ID).update(Password=Password)
            Employee.objects.filter(ID=ID).update(Sex=Sex)
            Employee.objects.filter(ID=ID).update(Age=Age)
            Employee.objects.filter(ID=ID).update(Phone=Phone)
            Employee.objects.filter(ID=ID).update(IsActive=IsActive)
            return JsonResponse({'code': 20000, 'message': '成功修改'})

    elif role == 'volunteer':
        UserName = json_param.get('UserName')
        Password = json_param.get('Password')
        Sex = json_param.get('Sex')
        Age = json_param.get('Age')
        Phone = json_param.get('Phone')
        ImgUrl = json_param.get('ImgUrl')
        IsActive = json_param.get('IsActive')

        results = Volunteer.objects.filter(UserName=UserName)
        if results.exists():
            for volunteer in results:
                if volunteer.ID != ID:
                    return JsonResponse({'code': 20003, 'message': '用户名已被注册'})
                else:
                    Volunteer.objects.filter(ID=ID).update(UserName=UserName)
                    Volunteer.objects.filter(ID=ID).update(Password=Password)
                    Volunteer.objects.filter(ID=ID).update(Sex=Sex)
                    Volunteer.objects.filter(ID=ID).update(Age=Age)
                    Volunteer.objects.filter(ID=ID).update(Phone=Phone)
                    Volunteer.objects.filter(ID=ID).update(ImgUrl=ImgUrl)
                    Volunteer.objects.filter(ID=ID).update(IsActive=IsActive)
                    return JsonResponse({'code': 20000, 'message': '成功修改'})

        results1 = Volunteer.objects.filter(Phone=Phone)
        if results1.exists():
            for volunteer in results1:
                if volunteer.ID != ID:
                    return JsonResponse({'code': 20003, 'message': '手机号已被注册'})
                else:

                    Volunteer.objects.filter(ID=ID).update(UserName=UserName)
                    Volunteer.objects.filter(ID=ID).update(Password=Password)
                    Volunteer.objects.filter(ID=ID).update(Sex=Sex)
                    Volunteer.objects.filter(ID=ID).update(Age=Age)
                    Volunteer.objects.filter(ID=ID).update(Phone=Phone)
                    Volunteer.objects.filter(ID=ID).update(ImgUrl=ImgUrl)
                    Volunteer.objects.filter(ID=ID).update(IsActive=IsActive)
                    return JsonResponse({'code': 20000, 'message': '成功修改'})

        if not (results1.exists() or results.exists()):
            Volunteer.objects.filter(ID=ID).update(UserName=UserName)
            Volunteer.objects.filter(ID=ID).update(Password=Password)
            Volunteer.objects.filter(ID=ID).update(Sex=Sex)
            Volunteer.objects.filter(ID=ID).update(Age=Age)
            Volunteer.objects.filter(ID=ID).update(Phone=Phone)
            Volunteer.objects.filter(ID=ID).update(ImgUrl=ImgUrl)
            Volunteer.objects.filter(ID=ID).update(IsActive=IsActive)
            return JsonResponse({'code': 20000, 'message': '成功修改'})


def employeeList(request):
    # print(1111111111111)
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})

    # print(token)
    # print(token)
    # print(result)
    # print(result.get('username'))
    role = getRoleByToken(token)
    # print(role)

    if role != "admin":
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    searchName = request.GET.get('UserName')
    searchPhone = request.GET.get('Phone')

    if not searchName and not searchPhone:
        employees = Employee.objects.all()
        total = Employee.objects.count()
    elif searchName and searchPhone:
        employees = Employee.objects.filter(UserName=searchName, Phone=searchPhone)
        total = Employee.objects.filter(UserName=searchName, Phone=searchPhone).count()
    elif not searchName and searchPhone:
        employees = Employee.objects.filter(Phone=searchPhone)
        total = Employee.objects.filter(Phone=searchPhone).count()
    else:
        employees = Employee.objects.filter(UserName=searchName)
        total = Employee.objects.filter(UserName=searchName).count()

    employee_list = []
    for employee in employees:
        employee_list.append(
            {
                'ID': employee.ID,
                'UserName': employee.UserName,
                'Password': employee.Password,
                'Sex': employee.Sex,
                'Age': employee.Age,
                'Phone': employee.Phone,
                'IsActive': employee.IsActive,
                'Created': employee.Created,
                'Updated': employee.Updated
            }
        )

    return JsonResponse({'code': 20000,
                         'message': 'success',
                         'data': {'total': total,
                                  'rows': employee_list}})


def employeeAdd(request):
    post_body = request.body
    # print(post_body)
    json_param = json.loads(post_body.decode())
    # print(json_param)
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    role = getRoleByToken(token)
    if role != "admin":
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    UserName = json_param.get('UserName')
    Password = json_param.get('Password')
    Sex = json_param.get('Sex')
    Age = json_param.get('Age')
    Phone = json_param.get('Phone')

    result = Employee.objects.filter(UserName=UserName)
    if result.exists():
        return JsonResponse({'code': 20002, 'message': '用户已存在'})

    result1 = Employee.objects.filter(Phone=Phone)
    if result1.exists():
        return JsonResponse({'code': 20003, 'message': '手机号已被注册'})

    if not result.exists() and not result1.exists():
        employee = Employee(UserName=UserName,
                            Password=Password,
                            Sex=Sex,
                            Age=Age,
                            Phone=Phone)
        employee.save()

        return JsonResponse({'code': 20000, 'message': '成功添加'})


def employeeDelete(request):
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    role = getRoleByToken(token)
    if role != "admin":
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    ID = request.GET.get('ID')
    print(request.GET)
    deleteEmployee = Employee.objects.get(ID=ID)
    # print(deleteUser)
    deleteEmployee.delete()
    return JsonResponse({'code': 20000, 'message': '删除成功'})


def employeeUpdate(request):
    post_body = request.body
    json_param = json.loads(post_body.decode())
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    role = getRoleByToken(token)
    if role != "admin":
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    ID = json_param.get('ID')
    UserName = json_param.get('UserName')
    Password = json_param.get('Password')
    Sex = json_param.get('Sex')
    Age = json_param.get('Age')
    Phone = json_param.get('Phone')
    IsActive = json_param.get('IsActive')

    results = Employee.objects.filter(UserName=UserName)
    if results.exists():
        for employee in results:
            if employee.ID != ID:
                return JsonResponse({'code': 20003, 'message': '用户名已被注册'})
            else:
                Employee.objects.filter(ID=ID).update(UserName=UserName)
                Employee.objects.filter(ID=ID).update(Password=Password)
                Employee.objects.filter(ID=ID).update(Sex=Sex)
                Employee.objects.filter(ID=ID).update(Age=Age)
                Employee.objects.filter(ID=ID).update(Phone=Phone)
                Employee.objects.filter(ID=ID).update(IsActive=IsActive)
                return JsonResponse({'code': 20000, 'message': '成功修改'})

    results1 = Employee.objects.filter(Phone=Phone)
    if results1.exists():
        for employee in results1:
            if employee.ID != ID:
                return JsonResponse({'code': 20003, 'message': '手机号已被注册'})
            else:

                Employee.objects.filter(ID=ID).update(UserName=UserName)
                Employee.objects.filter(ID=ID).update(Password=Password)
                Employee.objects.filter(ID=ID).update(Sex=Sex)
                Employee.objects.filter(ID=ID).update(Age=Age)
                Employee.objects.filter(ID=ID).update(Phone=Phone)
                Employee.objects.filter(ID=ID).update(IsActive=IsActive)
                return JsonResponse({'code': 20000, 'message': '成功修改'})

    if not (results1.exists() or results.exists()):
        Employee.objects.filter(ID=ID).update(UserName=UserName)
        Employee.objects.filter(ID=ID).update(Password=Password)
        Employee.objects.filter(ID=ID).update(Sex=Sex)
        Employee.objects.filter(ID=ID).update(Age=Age)
        Employee.objects.filter(ID=ID).update(Phone=Phone)
        Employee.objects.filter(ID=ID).update(IsActive=IsActive)
        return JsonResponse({'code': 20000, 'message': '成功修改'})


def employeeDetailByID(request):
    # print(11111111111)
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    role = getRoleByToken(token)
    if role != "admin":
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    ID = request.GET.get('ID')
    # print(ID)

    result = Employee.objects.filter(ID=ID).first()
    return JsonResponse({
        "code": 20000,
        "data": {
            'ID': result.ID,
            'UserName': result.UserName,
            'Password': result.Password,
            'Sex': result.Sex,
            'Age': result.Age,
            'Phone': result.Phone,
            'IsActive': result.IsActive,
            'Created': result.Created,
            'Updated': result.Updated
        }
    })


def volunteerList(request):
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    # print(result)
    # print(result.get('username'))
    role = getRoleByToken(token)

    if role == 'volunteer':
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    searchName = request.GET.get('UserName')
    searchPhone = request.GET.get('Phone')
    if not searchName and not searchPhone:
        volunteers = Volunteer.objects.all()
        total = Volunteer.objects.count()
    elif searchName and searchPhone:
        volunteers = Volunteer.objects.filter(UserName=searchName, Phone=searchPhone)
        total = Volunteer.objects.filter(UserName=searchName, Phone=searchPhone).count()
    elif not searchName and searchPhone:
        volunteers = Volunteer.objects.filter(Phone=searchPhone)
        total = Volunteer.objects.filter(Phone=searchPhone).count()
    else:
        volunteers = Volunteer.objects.filter(UserName=searchName)
        total = Volunteer.objects.filter(UserName=searchName).count()

    volunteer_list = []
    for volunteer in volunteers:
        url = getImgUrl(volunteer.ImgUrl)

        volunteer_list.append(
            {
                'ID': volunteer.ID,
                'UserName': volunteer.UserName,
                'Password': volunteer.Password,
                'Sex': volunteer.Sex,
                'Age': volunteer.Age,
                'Phone': volunteer.Phone,
                'ImgUrl': volunteer.ImgUrl,
                'Url': url,
                'IsActive': volunteer.IsActive,
                'Created': volunteer.Created,
                'Updated': volunteer.Updated
            }
        )

    return JsonResponse({'code': 20000,
                         'message': 'success',
                         'data': {'total': total,
                                  'rows': volunteer_list}})


def volunteerAdd(request):
    post_body = request.body
    # print(post_body)
    json_param = json.loads(post_body.decode())
    # print(json_param)
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    # print(result)
    # print(result.get('username'))
    role = getRoleByToken(token)

    if role == 'volunteer':
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    UserName = json_param.get('UserName')
    Password = json_param.get('Password')
    Sex = json_param.get('Sex')
    Age = json_param.get('Age')
    Phone = json_param.get('Phone')
    ImgUrl = json_param.get('ImgUrl')
    result = Volunteer.objects.filter(UserName=UserName)
    if result.exists():
        return JsonResponse({'code': 20002, 'message': '用户已存在'})

    result1 = Volunteer.objects.filter(Phone=Phone)
    if result1.exists():
        return JsonResponse({'code': 20003, 'message': '手机号已被注册'})

    if not result.exists() and not result1.exists():
        volunteer = Volunteer(UserName=UserName,
                              Password=Password,
                              Sex=Sex,
                              Age=Age,
                              Phone=Phone,
                              ImgUrl=ImgUrl)
        volunteer.save()

        return JsonResponse({'code': 20000, 'message': '成功添加'})


def volunteerDelete(request):
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    # print(result)
    # print(result.get('username'))
    role = getRoleByToken(token)

    if role == 'volunteer':
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    ID = request.GET.get('ID')
    deleteVolunteer = Volunteer.objects.get(ID=ID)
    # print(deleteUser)
    deleteVolunteer.delete()
    return JsonResponse({'code': 20000, 'message': '删除成功'})


def volunteerUpdate(request):
    post_body = request.body
    json_param = json.loads(post_body.decode())
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    # print(result)
    # print(result.get('username'))
    role = getRoleByToken(token)

    if role == 'volunteer':
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    ID = json_param.get('ID')
    UserName = json_param.get('UserName')
    Password = json_param.get('Password')
    Sex = json_param.get('Sex')
    Age = json_param.get('Age')
    Phone = json_param.get('Phone')
    ImgUrl = json_param.get('ImgUrl')
    IsActive = json_param.get('IsActive')

    results = Volunteer.objects.filter(UserName=UserName)
    if results.exists():
        for volunteer in results:
            if volunteer.ID != ID:
                return JsonResponse({'code': 20003, 'message': '用户名已被注册'})
            else:
                Volunteer.objects.filter(ID=ID).update(UserName=UserName)
                Volunteer.objects.filter(ID=ID).update(Password=Password)
                Volunteer.objects.filter(ID=ID).update(Sex=Sex)
                Volunteer.objects.filter(ID=ID).update(Age=Age)
                Volunteer.objects.filter(ID=ID).update(Phone=Phone)
                Volunteer.objects.filter(ID=ID).update(ImgUrl=ImgUrl)
                Volunteer.objects.filter(ID=ID).update(IsActive=IsActive)
                return JsonResponse({'code': 20000, 'message': '成功修改'})

    results1 = Volunteer.objects.filter(Phone=Phone)
    if results1.exists():
        for volunteer in results1:
            if volunteer.ID != ID:
                return JsonResponse({'code': 20003, 'message': '手机号已被注册'})
            else:

                Volunteer.objects.filter(ID=ID).update(UserName=UserName)
                Volunteer.objects.filter(ID=ID).update(Password=Password)
                Volunteer.objects.filter(ID=ID).update(Sex=Sex)
                Volunteer.objects.filter(ID=ID).update(Age=Age)
                Volunteer.objects.filter(ID=ID).update(Phone=Phone)
                Volunteer.objects.filter(ID=ID).update(ImgUrl=ImgUrl)
                Volunteer.objects.filter(ID=ID).update(IsActive=IsActive)
                return JsonResponse({'code': 20000, 'message': '成功修改'})

    if not (results1.exists() or results.exists()):
        Volunteer.objects.filter(ID=ID).update(UserName=UserName)
        Volunteer.objects.filter(ID=ID).update(Password=Password)
        Volunteer.objects.filter(ID=ID).update(Sex=Sex)
        Volunteer.objects.filter(ID=ID).update(Age=Age)
        Volunteer.objects.filter(ID=ID).update(Phone=Phone)
        Volunteer.objects.filter(ID=ID).update(ImgUrl=ImgUrl)
        Volunteer.objects.filter(ID=ID).update(IsActive=IsActive)
        return JsonResponse({'code': 20000, 'message': '成功修改'})


def volunteerDetailByID(request):
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    role = getRoleByToken(token)
    if role == "volunteer":
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    ID = request.GET.get('ID')

    result = Volunteer.objects.filter(ID=ID).first()
    url = getImgUrl(result.ImgUrl)
    return JsonResponse({
        "code": 20000,
        "data": {
            'ID': result.ID,
            'UserName': result.UserName,
            'Password': result.Password,
            'Sex': result.Sex,
            'Age': result.Age,
            'Phone': result.Phone,
            'ImgUrl': result.ImgUrl,
            'Url': url,
            'IsActive': result.IsActive,
            'Created': result.Created,
            'Updated': result.Updated
        }
    })


# elderly
def elderlyList(request):
    # token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    # if token and token.startswith("Bearer "):
    #     token = token.split(" ")[1]  # 提取实际的 token 部分
    # else:
    #     return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    # print(result)
    # print(result.get('username'))
    searchName = request.GET.get('UserName')
    searchPhone = request.GET.get('Phone')
    if not searchName and not searchPhone:
        elderlys = Elderly.objects.all()
        total = Elderly.objects.count()
    elif searchName and searchPhone:
        elderlys = Elderly.objects.filter(UserName=searchName, Phone=searchPhone)
        total = Elderly.objects.filter(UserName=searchName, Phone=searchPhone).count()
    elif not searchName and searchPhone:
        elderlys = Elderly.objects.filter(Phone=searchPhone)
        total = Elderly.objects.filter(Phone=searchPhone).count()
    else:
        elderlys = Elderly.objects.filter(UserName=searchName)
        total = Elderly.objects.filter(UserName=searchName).count()

    elderly_list = []
    for elderly in elderlys:
        url = getImgUrl(elderly.ImgUrl)
        elderly_list.append(
            {
                'ID': elderly.ID,
                'UserName': elderly.UserName,

                'Sex': elderly.Sex,
                'Age': elderly.Age,
                'Birthday': elderly.Birthday,
                'Phone': elderly.Phone,
                'Healthy': elderly.Healthy,
                'GuardianName': elderly.GuardianName,
                'GuardianPhone': elderly.GuardianPhone,
                'ImgUrl': elderly.ImgUrl,
                'Url': url,
                'IsActive': elderly.IsActive,
                'Created': elderly.Created,
                'Updated': elderly.Updated
            }
        )

    return JsonResponse({'code': 20000,
                         'message': 'success',
                         'data': {'total': total,
                                  'rows': elderly_list}})


def elderlyAdd(request):
    post_body = request.body
    # print(post_body)
    json_param = json.loads(post_body.decode())
    # print(json_param)
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    # print(result)
    # print(result.get('username'))
    role = getRoleByToken(token)

    if role == 'volunteer':
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    UserName = json_param.get('UserName')

    Sex = json_param.get('Sex')
    Age = json_param.get('Age')
    Birthday = json_param.get('Birthday')
    Phone = json_param.get('Phone')
    Healthy = json_param.get('Healthy')
    GuardianName = json_param.get('GuardianName')
    GuardianPhone = json_param.get('GuardianPhone')
    ImgUrl = json_param.get('ImgUrl')
    result = Elderly.objects.filter(UserName=UserName)
    if result.exists():
        return JsonResponse({'code': 20002, 'message': '用户已存在'})

    result1 = Elderly.objects.filter(Phone=Phone)
    if result1.exists():
        return JsonResponse({'code': 20003, 'message': '手机号已被注册'})

    if not result.exists() and not result1.exists():
        elderly = Elderly(UserName=UserName,
                          Sex=Sex,
                          Age=Age,
                          Birthday=Birthday,
                          Phone=Phone,
                          Healthy=Healthy,
                          GuardianName=GuardianName,
                          GuardianPhone=GuardianPhone,
                          ImgUrl=ImgUrl)
        elderly.save()

        return JsonResponse({'code': 20000, 'message': '成功添加'})


def elderlyDelete(request):
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    # print(result)
    # print(result.get('username'))
    role = getRoleByToken(token)

    if role == 'volunteer':
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    ID = request.GET.get('ID')
    deleteElderly = Elderly.objects.get(ID=ID)
    # print(deleteUser)
    deleteElderly.delete()
    return JsonResponse({'code': 20000, 'message': '删除成功'})


def elderlyUpdate(request):
    post_body = request.body
    json_param = json.loads(post_body.decode())
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    # print(result)
    # print(result.get('username'))
    role = getRoleByToken(token)

    if role == 'volunteer':
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    ID = json_param.get('ID')
    UserName = json_param.get('UserName')
    # print(UserName)
    Sex = json_param.get('Sex')
    Age = json_param.get('Age')
    Birthday = json_param.get('Birthday')
    Phone = json_param.get('Phone')
    Healthy = json_param.get('Healthy')
    GuardianName = json_param.get('GuardianName')
    GuardianPhone = json_param.get('GuardianPhone')
    ImgUrl = json_param.get('ImgUrl')
    IsActive = json_param.get('IsActive')

    results = Elderly.objects.filter(UserName=UserName)
    if results.exists():
        for elderly in results:
            if elderly.ID != ID:
                return JsonResponse({'code': 20003, 'message': '用户名已被注册'})
            else:
                Elderly.objects.filter(ID=ID).update(UserName=UserName)
                Elderly.objects.filter(ID=ID).update(Sex=Sex)
                Elderly.objects.filter(ID=ID).update(Age=Age)
                Elderly.objects.filter(ID=ID).update(Birthday=Birthday)
                Elderly.objects.filter(ID=ID).update(Phone=Phone)
                Elderly.objects.filter(ID=ID).update(Healthy=Healthy)
                Elderly.objects.filter(ID=ID).update(GuardianName=GuardianName)
                Elderly.objects.filter(ID=ID).update(GuardianPhone=GuardianPhone)
                Elderly.objects.filter(ID=ID).update(ImgUrl=ImgUrl)
                Elderly.objects.filter(ID=ID).update(IsActive=IsActive)
                return JsonResponse({'code': 20000, 'message': '成功修改'})

    results1 = Elderly.objects.filter(Phone=Phone)
    if results1.exists():
        for elderly in results1:
            if elderly.ID != ID:
                return JsonResponse({'code': 20003, 'message': '手机号已被注册'})
            else:
                Elderly.objects.filter(ID=ID).update(UserName=UserName)
                Elderly.objects.filter(ID=ID).update(Sex=Sex)
                Elderly.objects.filter(ID=ID).update(Age=Age)
                Elderly.objects.filter(ID=ID).update(Birthday=Birthday)
                Elderly.objects.filter(ID=ID).update(Phone=Phone)
                Elderly.objects.filter(ID=ID).update(Healthy=Healthy)
                Elderly.objects.filter(ID=ID).update(GuardianName=GuardianName)
                Elderly.objects.filter(ID=ID).update(GuardianPhone=GuardianPhone)
                Elderly.objects.filter(ID=ID).update(ImgUrl=ImgUrl)
                Elderly.objects.filter(ID=ID).update(IsActive=IsActive)
                return JsonResponse({'code': 20000, 'message': '成功修改'})

    if not (results1.exists() or results.exists()):
        Elderly.objects.filter(ID=ID).update(UserName=UserName)
        Elderly.objects.filter(ID=ID).update(Sex=Sex)
        Elderly.objects.filter(ID=ID).update(Age=Age)
        Elderly.objects.filter(ID=ID).update(Birthday=Birthday)
        Elderly.objects.filter(ID=ID).update(Phone=Phone)
        Elderly.objects.filter(ID=ID).update(Healthy=Healthy)
        Elderly.objects.filter(ID=ID).update(GuardianName=GuardianName)
        Elderly.objects.filter(ID=ID).update(GuardianPhone=GuardianPhone)
        Elderly.objects.filter(ID=ID).update(ImgUrl=ImgUrl)
        Elderly.objects.filter(ID=ID).update(IsActive=IsActive)
        return JsonResponse({'code': 20000, 'message': '成功修改'})


def elderlyDetailByID(request):
    # token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    # if token and token.startswith("Bearer "):
    #     token = token.split(" ")[1]  # 提取实际的 token 部分
    # else:
    #     return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    # role = getRoleByToken(token)
    # if role == "volunteer":
    #     return JsonResponse({'code': 20003, 'message': '你没有权限'})
    ID = request.GET.get('ID')
    result = Elderly.objects.filter(ID=ID).first()
    url = getImgUrl(result.ImgUrl)
    return JsonResponse({
        "code": 20000,
        "data": {
            'ID': result.ID,
            'UserName': result.UserName,
            'Sex': result.Sex,
            'Age': result.Age,
            'Birthday': result.Birthday,
            'Phone': result.Phone,
            'Healthy': result.Healthy,
            'GuardianName': result.GuardianName,
            'GuardianPhone': result.GuardianPhone,
            'ImgUrl': result.ImgUrl,
            'Url': url,
            'IsActive': result.IsActive,
            'Created': result.Created,
            'Updated': result.Updated
        }
    })


def uploadCloud(request):
    print(request.FILES)
    file = request.FILES['file']
    # 填写上传到OSS后的文件名
    oss_file_name = 'old-care/{}'.format(file.name)
    # 上传文件
    bucket.put_object(oss_file_name, file.read())
    print(111111111111111111111111111)
    # 获取文件的URL
    # image_url = f'https://{bucket_name}.{endpoint.split("//")[1]}/{oss_file_name}'
    # url = bucket.sign_url('GET', 'old-care/volunteer/image.jpg', 10 * 60)
    # print(url)
    # print(f'Image URL: {image_url}')
    return JsonResponse({'code': 20000, 'message': '成功上传', 'data': oss_file_name})
