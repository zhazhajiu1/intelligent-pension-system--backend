from __future__ import unicode_literals
from django.db import models


# Create your models here.
class Admin(models.Model):
    objects = None
    ID = models.AutoField(primary_key=True)
    UserName = models.CharField(max_length=50, unique=True)
    Password = models.CharField(max_length=50)
    Sex = models.CharField(max_length=20)
    Phone = models.CharField(max_length=50)
    IsActive = models.CharField(max_length=10, default='0')
    Created = models.DateTimeField(auto_now_add=True)
    Updated = models.DateTimeField(auto_now=True)

    def __unicode__(self):
        return u'Admin:%s' % self.ID


class Employee(models.Model):
    objects = None
    ID = models.AutoField(primary_key=True)
    UserName = models.CharField(max_length=50, unique=True)
    Password = models.CharField(max_length=50)
    Sex = models.CharField(max_length=20)
    Age = models.IntegerField()
    Phone = models.CharField(max_length=50)
    IsActive = models.CharField(max_length=10, default='0')
    Created = models.DateTimeField(auto_now_add=True)
    Updated = models.DateTimeField(auto_now=True)

    def __unicode__(self):
        return u'Employee:%s' % self.ID


class Volunteer(models.Model):
    objects = None
    ID = models.AutoField(primary_key=True)
    UserName = models.CharField(max_length=50, unique=True)
    Password = models.CharField(max_length=50)
    Sex = models.CharField(max_length=20)
    Age = models.IntegerField()
    Phone = models.CharField(max_length=50)
    ImgUrl = models.CharField(max_length=255)
    IsActive = models.CharField(max_length=10, default='0')
    Created = models.DateTimeField(auto_now_add=True)
    Updated = models.DateTimeField(auto_now=True)

    def __unicode__(self):
        return u'Volunteer:%s' % self.ID


class Elderly(models.Model):
    objects = None
    ID = models.AutoField(primary_key=True)
    UserName = models.CharField(max_length=50, unique=True)
    Sex = models.CharField(max_length=20)
    Age = models.IntegerField()
    Birthday = models.CharField(max_length=50, default='1960-01-01')
    Phone = models.CharField(max_length=50)
    Healthy = models.CharField(max_length=50)
    GuardianName = models.CharField(max_length=50)
    GuardianPhone = models.CharField(max_length=50)
    ImgUrl = models.CharField(max_length=255)
    IsActive = models.CharField(max_length=10, default='0')
    Created = models.DateTimeField(auto_now_add=True)
    Updated = models.DateTimeField(auto_now=True)

    def __unicode__(self):
        return u'Elderly:%s' % self.ID
