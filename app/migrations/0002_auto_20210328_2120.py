# Generated by Django 3.1.7 on 2021-03-28 18:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Doc',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('upload', models.ImageField(upload_to='images/')),
            ],
        ),
        migrations.DeleteModel(
            name='Photo',
        ),
    ]