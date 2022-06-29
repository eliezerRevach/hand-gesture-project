
import os
import random
from shutil import copyfile


path_folder_input="from_scissors_model_data_2d_p"
path_output_folder='from_scissors_model_data_2d_p_train_test'
percent=0.85
def img_train_test_split(file_source_dir, train_size):
    """
    Randomly splits images over a train and validation folder, while preserving the folder structure

    Parameters
    ----------
    file_source_dir : string
        Path to the folder with the images to be split. Can be absolute or relative path   

    train_size : float
        Proportion of the original images that need to be copied in the subdirectory in the train folder
    """
    if not (isinstance(file_source_dir, str)):
        raise AttributeError('img_source_dir must be a string')

    if not os.path.exists(file_source_dir):
        raise OSError('img_source_dir does not exist')

    if not (isinstance(train_size, float)):
        raise AttributeError('train_size must be a float')

    # Set up empty folder structure if not exists
    if not os.path.exists(path_output_folder):
        os.makedirs(path_output_folder)
    else:
        if not os.path.exists(path_output_folder+'/train'):
            os.makedirs(path_output_folder+'/train')
        if not os.path.exists(path_output_folder+'/validation'):
            os.makedirs(path_output_folder+'/validation')

    # Get the subdirectories in the main file folder
    subdirs = [subdir for subdir in os.listdir(file_source_dir) if os.path.isdir(os.path.join(file_source_dir, subdir))]

    for subdir in subdirs:
        subdir_fullpath = os.path.join(file_source_dir, subdir)
        if len(os.listdir(subdir_fullpath)) == 0:
            print(subdir_fullpath + ' is empty')
            break

        train_subdir = os.path.join(path_output_folder+'/train', subdir)
        validation_subdir = os.path.join(path_output_folder+'/validation', subdir)

        # Create subdirectories in train and validation folders
        if not os.path.exists(train_subdir):
            os.makedirs(train_subdir)

        if not os.path.exists(validation_subdir):
            os.makedirs(validation_subdir)

        train_counter = 0
        validation_counter = 0

        # Randomly assign an file to train or validation folder
        for filename in os.listdir(subdir_fullpath):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".csv") or filename.endswith(".txt"):
                fileparts = filename.split('.')

                if random.uniform(0, 1) <= train_size:
                    copyfile(os.path.join(subdir_fullpath, filename),
                             os.path.join(train_subdir, str(train_counter) + '.' + fileparts[1]))
                    train_counter += 1
                else:
                    copyfile(os.path.join(subdir_fullpath, filename),
                             os.path.join(validation_subdir, str(validation_counter) + '.' + fileparts[1]))
                    validation_counter += 1

        print('Copied ' + str(train_counter) + ' images to data/train/' + subdir)
        print('Copied ' + str(validation_counter) + ' images to data/validation/' + subdir)



img_train_test_split(path_folder_input+"/", percent)