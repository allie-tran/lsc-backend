from zipfile import ZipFile
import zipfile
import os, json, shutil


def unzip_uploaded_file(source_path, destination_path):
    """
    Parameters:
    - source_path: path of the uploaded zip file
    - destination_path: the path to which data in zip file are extracted
    Return:
    - True if success / False if there is any error
    """
    if zipfile.is_zipfile(source_path):
        with ZipFile(source_path, 'r') as zip_obj:
            zip_obj.extractall(destination_path)
        refined_extracted_structure(destination_path)
        return True
    return False


def refined_extracted_structure(destination_path):
    """
    Parameters:
    - destination_path: the path to which data in zip file are extracted
    """
    if os.path.exists(os.path.join(destination_path, '__MACOSX')):
        shutil.rmtree(os.path.join(destination_path, '__MACOSX'))


def check_zip_file(source_path):
    """
    Parameters:
    - source_path: path of the uploaded zip file
    Return:
    - True if success / False if there is any error
    """
    validated = zipfile.is_zipfile(source_path)
    # if validated == False:
    #     return validated
    # with ZipFile(source_path, 'r') as f:
    #     if f.testzip() != None:
    #         validated = False
    return validated


def remove_uploaded_zip_file(source_path):
    """
    Parameters:
    - source_path: path of the uploaded zip file
    """
    os.remove(source_path)


def dump_data_structure(source_path):
    """
    Parameters:
    - source_path: path to the extracted folder. The structure should be: 
        user_id
            |__date
                |__image_id
    Return:
    - user_schema: schema that contains User Lifelog Information.
    - image_annotation_schema: schema that contains detailed information and annotation of each image 
    ** Note: Always sort data in temporal order due to MongoDB insertion
    """
    user_id = [d for d in os.listdir(source_path) if not d == '.DS_Store'][0] # Prevent .DS_Store file from MACOS computer
    source_path = os.path.join(source_path, user_id)
    dates = [d for d in sorted(os.listdir(source_path)) if not d == '.DS_Store'] # Prevent .DS_Store file from MACOS computer
    overall_data = []
    detailed_data = []
    for d in dates:
        source_date_path = os.path.join(source_path, d)
        image_name_list = [f for f in sorted(os.listdir(source_date_path)) if not f == '.DS_Store'] # Prevent .DS_Store file from MACOS computer
        image_paths = [os.path.join(user_id, d, image_name) for image_name in image_name_list]
        overall_data.append({
            'date': d,
            'image_id': image_name_list
        })
        for img_path in image_paths:
            detailed_data.append({
                '_id': img_path,
                'image_id': img_path,
                'image_path': img_path,
                'date': d,
                'user_id': user_id,
                'annotator_id': None,
                'annotated': False,
                'annotation': {
                    'activities': {
                        'cooking': False,
                        'eating': False,
                        'drinking': False,
                        'gardening': False,
                        'housework': False,
                        'screen_time': False,
                        'reading': False,
                        'hygiene_in_bathroom': False,
                        'physical_activities': False,
                        'socializing': False,
                        'sedentary': False,
                        'shopping': False,
                    },
                    'locations': {
                        'indoor': False,
                        'outdoor': False,
                        'home': False,
                    },
                    'consumptions': {
                        'food': False,
                        'beverages': False, 
                        'medications': False,
                    }
                },
            })
    user_schema = [{ 
        '_id': user_id,
        'user_id': user_id,
        'data': overall_data
    }]
    image_annotation_schema = detailed_data
    return user_schema, image_annotation_schema