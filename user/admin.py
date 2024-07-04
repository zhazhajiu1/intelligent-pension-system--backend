from __future__ import unicode_literals

# Register your models here.
# user.site.re
from django.contrib import admin
from .models import Admin
from .models import Employee
from .models import Volunteer
from .models import Elderly

# Register your models here.
admin.site.register([Admin])
admin.site.register([Employee])
admin.site.register([Volunteer])
admin.site.register([Elderly])