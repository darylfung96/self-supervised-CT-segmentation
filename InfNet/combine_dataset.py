import os
from natsort import natsorted
from shutil import copyfile
lunginfection = ['Edge', 'GT', 'Imgs']
multiinfection = ['GT', 'Imgs', 'Prior']
sets = ['TrainingSet', 'TestingSet', 'ValSet']
set_name = ['Train', 'Test', 'Val']

# LungInfection
# make dir first
for current_type in lunginfection:
    new_dir = f'Dataset/AllSet/LungInfection-All/{current_type}'
    os.makedirs(new_dir, exist_ok=True)
# start copying files
current_index = 0
current_type = lunginfection[0]
new_dir = f'Dataset/AllSet/LungInfection-All/{current_type}'
for set_index in range(len(sets)):
    current_dir = f'Dataset/{sets[set_index]}/LungInfection-{set_name[set_index]}'

    training_data_dir = os.path.join(current_dir, current_type)
    all_items = natsorted(os.listdir(training_data_dir))
    for item in all_items:
        item_format = item.split('.')
        item_format[0] = str(current_index)
        new_name = '.'.join(item_format)
        current_index += 1

        # make sure the other files exist before copying (same filename)
        other_files_exist = True
        for other_type in lunginfection[1:]:
            testing_data_dir = os.path.join(current_dir, other_type)
            other_item_format = os.listdir(testing_data_dir)[0].split('.')[1]
            other_dir = f'Dataset/AllSet/LungInfection-All/{other_type}'
            if not os.path.exists(os.path.join(testing_data_dir, f'{str(current_index)}.{other_item_format}')):
                other_files_exist = False
                continue
        if not other_files_exist:
            continue

        copyfile(os.path.join(training_data_dir, item), os.path.join(new_dir, new_name))
        for other_type in lunginfection[1:]:
            testing_data_dir = os.path.join(current_dir, other_type)
            other_item_format = os.listdir(testing_data_dir)[0].split('.')[1]
            other_dir = f'Dataset/AllSet/LungInfection-All/{other_type}'
            copyfile(os.path.join(testing_data_dir, f'{str(current_index)}.{other_item_format}'),
                     os.path.join(other_dir, f'{str(current_index)}.{other_item_format}'))

# MultiClassInfection
# make dir first
for current_type in lunginfection:
    new_dir = f'Dataset/AllSet/MultiClassInfection-All/{current_type}'
    os.makedirs(new_dir, exist_ok=True)
# start copying files
current_index = 0
current_type = lunginfection[0]
new_dir = f'Dataset/AllSet/MultiClassInfection-All/{current_type}'
for set_index in range(len(sets)):
    current_dir = f'Dataset/{sets[set_index]}/MultiClassInfection-{set_name[set_index]}'

    training_data_dir = os.path.join(current_dir, current_type)
    all_items = natsorted(os.listdir(training_data_dir))
    for item in all_items:
        item_format = item.split('.')
        item_format[0] = str(current_index)
        new_name = '.'.join(item_format)
        current_index += 1

        # make sure the other files exist before copying (same filename)
        other_files_exist = True
        for other_type in lunginfection[1:]:
            testing_data_dir = os.path.join(current_dir, other_type)
            other_item_format = os.listdir(testing_data_dir)[0].split('.')[1]
            other_dir = f'Dataset/AllSet/MultiClassInfection-All/{other_type}'
            if not os.path.exists(os.path.join(testing_data_dir, f'{str(current_index)}.{other_item_format}')):
                other_files_exist = False
                continue
        if not other_files_exist:
            continue

        copyfile(os.path.join(training_data_dir, item), os.path.join(new_dir, new_name))
        for other_type in lunginfection[1:]:
            testing_data_dir = os.path.join(current_dir, other_type)
            other_dir = f'Dataset/AllSet/MultiClassInfection-All/{other_type}'
            copyfile(os.path.join(testing_data_dir, item), os.path.join(other_dir, new_name))
