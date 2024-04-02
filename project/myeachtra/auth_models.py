import base64
import json

import redis
from rest_framework_simplejwt.serializers import TokenObtainSerializer
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView

from .settings import REDIS_HOST, REDIS_PORT


class CustomTokenObtainPairSerializer(TokenObtainSerializer):

    @classmethod
    def get_token(cls, user):
        refresh = RefreshToken.for_user(user)
        return refresh

    def validate(self, attrs):
        data = super().validate(attrs)
        if self.user:
            refresh = self.get_token(self.user)
            data["refresh"] = str(refresh)
            data["access"] = str(refresh.token)
        self.update_redis_db(data)
        return data

    def update_redis_db(self, token_data):
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        access_token = token_data["access"]
        _, payload, _ = access_token.split(".")
        payload = json.loads(base64.b64decode(payload).decode("ascii"))
        user_id = payload["user_id"]
        r.set(f"user_id:{user_id}", access_token)


class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer
