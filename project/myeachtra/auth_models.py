import jwt
import bcrypt
import redis
from fastapi import HTTPException, Request
import os
from configs import REDIS_HOST, REDIS_PORT
from query_parse.types.requests import LoginRequest

SECRET = os.getenv("JWT_SECRET", "")
assert SECRET, "JWT_SECRET is not set"

def create_user(request: LoginRequest) -> None:
    """
    Create a new user
    """
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    if r.get(request.username):
        raise HTTPException(status_code=400, detail="User already exists")
    r.set(request.username, bcrypt.hashpw(request.password.encode(), bcrypt.gensalt()))
    print(f"User {request.username} created")

def generate_token(username: str) -> str:
    """
    Generate a token for the user
    """
    return jwt.encode({"username": username}, SECRET, algorithm="HS256")


def verify_user(request: LoginRequest) -> str:
    """
    Verify user credentials and return an access token
    """
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    hashed_password = r.get(request.username)
    if not hashed_password:
        raise HTTPException(status_code=401, detail="User does not exist")
    if bcrypt.checkpw(request.password.encode(), str(hashed_password).encode()):
        return generate_token(request.username)
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

def get_user(request: Request) -> str:
    """
    Get the user from the request to make sure the user is authenticated
    """
    token = request.headers.get("Authorization") # Bearer token
    if not token:
        raise HTTPException(status_code=401, detail="Please log in")
    return verify_token(token.split(" ")[1])
