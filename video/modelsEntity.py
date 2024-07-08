from __future__ import unicode_literals
from django.db import models


# Create your models here.
class Emotion(models.Model):
    objects = None
    ID = models.AutoField(primary_key=True)
    ElderlyName = models.CharField(max_length=50)
    ImgUrl = models.CharField(max_length=255)
    Created = models.DateTimeField(auto_now_add=True)
    Type = models.CharField(max_length=50)

    def __unicode__(self):
        return u'Emotion:%s' % self.ID


class Fall(models.Model):
    objects = None
    ID = models.AutoField(primary_key=True)
    ImgUrl = models.CharField(max_length=255)
    Created = models.DateTimeField(auto_now_add=True)

    def __unicode__(self):
        return u'Fall:%s' % self.ID


class Unknow(models.Model):
    objects = None
    ID = models.AutoField(primary_key=True)
    ImgUrl = models.CharField(max_length=255)
    Created = models.DateTimeField(auto_now_add=True)

    def __unicode__(self):
        return u'Unknow:%s' % self.ID


class Intrusion(models.Model):
    objects = None
    ID = models.AutoField(primary_key=True)
    ImgUrl = models.CharField(max_length=255)
    Created = models.DateTimeField(auto_now_add=True)

    def __unicode__(self):
        return u'Unknow:%s' % self.ID


class Reaction(models.Model):
    objects = None
    ID = models.AutoField(primary_key=True)
    ElderlyName = models.CharField(max_length=50)
    VolunteerName = models.CharField(max_length=50)
    ImgUrl = models.CharField(max_length=255)
    Created = models.DateTimeField(auto_now_add=True)

    def __unicode__(self):
        return u'Unknow:%s' % self.ID


class Fire(models.Model):
    objects = None
    ID = models.AutoField(primary_key=True)
    ImgUrl = models.CharField(max_length=255)
    Created = models.DateTimeField(auto_now_add=True)

    def __unicode__(self):
        return u'Fire:%s' % self.ID