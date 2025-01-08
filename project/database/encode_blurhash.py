import blurhash
from PIL import Image

from configs import IMAGE_DIRECTORY
from database.main import get_db, image_collection
from tqdm.auto import tqdm

from query_parse.types.requests import Data


THUMBNAIL_HEIGHT: int = 768 // 7 # 109

def encode_blurhash(image_path: str, aspect_ratio: float, x_components: int = 4, y_components: int = 3) -> str:
    image = Image.open(image_path)
    image.thumbnail((THUMBNAIL_HEIGHT * aspect_ratio, THUMBNAIL_HEIGHT))
    return blurhash.encode(image, x_components, y_components)

def batch_encode(data: Data = Data.LSC23):
    db = get_db(data)
    size = image_collection(db).count_documents({})
    for image in tqdm(image_collection(db).find(), total=size):
        image_path = image["image"]
        aspect_ratio = image["aspect_ratio"]
        hash_code = encode_blurhash(f"{IMAGE_DIRECTORY}/{image_path}", aspect_ratio)
        image_collection(db).update_one({"_id": image["_id"]}, {"$set": {"hash_code": hash_code}})
