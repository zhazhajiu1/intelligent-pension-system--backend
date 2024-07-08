from __future__ import unicode_literals

# Register your models here.
# user.site.re
from django.contrib import admin
from .modelsEntity import Emotion
from .modelsEntity import Fall
from .modelsEntity import Unknow
from .modelsEntity import Intrusion
from .modelsEntity import Reaction
from .modelsEntity import Fire

admin.site.register([Emotion])
admin.site.register([Fall])
admin.site.register([Unknow])
admin.site.register([Intrusion])
admin.site.register([Reaction])
admin.site.register([Fire])