from unittest.mock import patch

import socketio
from django.contrib.auth import get_user_model
from django.test import TestCase

from realtime import sio_app
from users.token_auth import issue_token


class SocketAuthTests(TestCase):
    def setUp(self):
        user = get_user_model().objects.create_user(username="sock", password="pass12345", role="GUARD", phone="0")
        self.key = issue_token(user).key

    def test_connect_without_token_is_refused(self):
        with self.assertRaises(socketio.exceptions.ConnectionRefusedError):
            sio_app.connect("sid-1", {}, None)

    def test_connect_with_token_from_auth_query_or_cookie(self):
        with patch.object(sio_app.sio, "save_session"), patch.object(sio_app.sio, "emit"):
            sio_app.connect("sid-2", {}, {"token": self.key})
            sio_app.connect("sid-3", {"QUERY_STRING": f"token={self.key}"}, None)
            sio_app.connect("sid-4", {"HTTP_COOKIE": f"tekeye_auth_token={self.key}"}, None)

    def test_emit_goes_to_rooms_only(self):
        with patch.object(sio_app.sio, "emit") as emit:
            sio_app.emit_invalidate(["hr", "attendance"])
        emit.assert_called_once()
        self.assertEqual(emit.call_args.kwargs["to"], ["hr", "attendance"])
