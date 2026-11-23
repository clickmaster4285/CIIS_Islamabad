import json
import logging

from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver

from .models import Staff
from .staff_photos import staff_photo_paths

logger = logging.getLogger(__name__)

# The "before" values travel on the instance itself. Module-level dicts keyed by
# pk / id() leaked entries when a save failed and could mix up two unsaved objects
# that reused the same id().
_BEFORE_ATTR = "_face_sync_before"


def _photos_fingerprint(staff: Staff) -> str:
    return json.dumps(staff_photo_paths(staff), sort_keys=False)


@receiver(pre_save, sender=Staff)
def cache_staff_profile_image(sender, instance: Staff, **kwargs):
    previous_image, previous_photos = "", "[]"
    if instance.pk:
        try:
            row = Staff.objects.filter(pk=instance.pk).values("staff_photos", "profile_image").first()
            if row:
                previous_image = str(row.get("profile_image") or "")
                prev_staff = Staff(staff_photos=row.get("staff_photos") or [], profile_image=row.get("profile_image"))
                previous_photos = _photos_fingerprint(prev_staff)
        except Exception:
            logger.debug("Could not read previous staff photos for %s", instance.pk, exc_info=True)
    setattr(instance, _BEFORE_ATTR, (previous_image, previous_photos))


@receiver(post_save, sender=Staff)
def enroll_staff_face_on_save(sender, instance: Staff, created: bool, **kwargs):
    previous_image, previous_photos = getattr(instance, _BEFORE_ATTR, ("", "[]"))
    try:
        delattr(instance, _BEFORE_ATTR)
    except AttributeError:
        pass
    current_image = str(instance.profile_image.name if instance.profile_image else "")
    current_photos = _photos_fingerprint(instance)

    if not staff_photo_paths(instance):
        return

    image_changed = created or (current_image != previous_image)
    photos_changed = created or (current_photos != previous_photos)
    if not image_changed and not photos_changed:
        return

    try:
        from .face_sync_async import schedule_staff_face_sync

        schedule_staff_face_sync(instance.pk, force=True)
    except Exception:
        logger.exception("Failed to schedule face enrollment for staff %s", instance.pk)
