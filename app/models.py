from django.db import models


# Create your models here.
class Doc(models.Model):
    upload = models.ImageField(upload_to='image/')
    Image_Name = models.CharField(max_length=30, blank=True)
    text = models.CharField(max_length=250, blank=True)

    def __str__(self):
        return str(self.pk)
