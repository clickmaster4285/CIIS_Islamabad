from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("employee_monitoring", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="employeecallrecord",
            name="device_uuid",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
        migrations.AddField(
            model_name="employeecallrecord",
            name="sim_slot",
            field=models.CharField(blank=True, default="", max_length=120),
        ),
    ]
