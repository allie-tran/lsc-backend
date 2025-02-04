import os
from typing import List

import bcrypt
import jwt
import redis
from configs import REDIS_HOST, REDIS_PORT
from database.main import user_collection
from fastapi import HTTPException, Request
from query_parse.types.requests import CreateUserRequest, Data, LoginRequest, LoginResponse

from myeachtra.dependencies import CamelCaseModel

SECRET = os.getenv("JWT_SECRET", "")
assert SECRET, "JWT_SECRET is not set"


class UserDetail(CamelCaseModel):
    username: str
    data_access: List[Data] = [Data.LSC23]


# Remove all redis keys
def flush_redis() -> None:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.flushall()
    print("Redis flushed")


# flush_redis()


def create_user(request: CreateUserRequest, overwrite=False) -> None:
    """
    Create a new user
    """
    if user_collection.find_one({"username": request.username}) and not overwrite:
        raise HTTPException(status_code=400, detail="User already exists")

    user_collection.update_one(
        {"username": request.username},
        {
            "$set": {
                "password": bcrypt.hashpw(request.password.encode(), bcrypt.gensalt()),
                "data_access": request.data_access,
            }
        },
        upsert=True,
    )
    print(f"User {request.username} created")


# Create some users now
def create_users() -> None:
    admin = CreateUserRequest(
        username=os.getenv("ADMIN_USERNAME", ""),
        password=os.getenv("ADMIN_PASSWORD", ""),
        data_access=[Data.LSC23, Data.Deakin],
    )
    lsc181 = CreateUserRequest(
        username=os.getenv("LSC_USERNAME", ""),
        password=os.getenv("LSC_PASSWORD", ""),
        data_access=[Data.LSC23],
    )
    tiens = CreateUserRequest(
        username=os.getenv("TIENS_USERNAME", ""),
        password=os.getenv("TIENS_PASSWORD", ""),
        data_access=[Data.LSC23],
    )
    create_user(admin, overwrite=True)
    create_user(lsc181)
    create_user(tiens)


# create_users()


def generate_token(username: str) -> str:
    """
    Generate a token for the user
    """
    return jwt.encode({"username": username}, SECRET, algorithm="HS256")

def find_user_by_username(username: str) -> UserDetail:
    """
    Find a user by username
    """
    user = user_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=401, detail="User does not exist")
    return UserDetail(**user)


def verify_user(request: LoginRequest) -> LoginResponse:
    """
    Verify user credentials and return an access token
    """
    user = user_collection.find_one({"username": request.username})
    if not user:
        raise HTTPException(status_code=401, detail="User does not exist")
    if bcrypt.checkpw(request.password.encode(), user["password"]):
        return LoginResponse(
            data_access=user["data_access"],
            session_id=generate_token(request.username),
        )
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")


def verify_token(token: str) -> str:
    """
    Verify the token and return the username
    """
    try:
        return jwt.decode(token, SECRET, algorithms=["HS256"])["username"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token is expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_user(request: Request) -> UserDetail:
    """
    Get the user from the request to make sure the user is authenticated
    """
    token = request.headers.get("Authorization")  # Bearer token
    if not token:
        raise HTTPException(status_code=401, detail="Please log in")
    username = verify_token(token.split(" ")[1])
    user = find_user_by_username(username)
    request_data = await request.json()
    verify_data_access(Data(request_data["data"]), user)
    return user


def verify_data_access(data: Data, user: UserDetail) -> None:
    """
    Verify the user has access to the data
    """
    if data not in user.data_access:
        raise HTTPException(
            status_code=403, detail="User does not have access to this data"
        )
