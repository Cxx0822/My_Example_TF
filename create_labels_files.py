# -*-coding:utf-8-*-
import os
import os.path
import yaml

with open("info.yml") as stream:
    my_data = yaml.load(stream, Loader=yaml.FullLoader) 
    # python3.6 可能要去掉Loader=yaml.FullLoader

my_labels = my_data['my_labels']
train_dir = my_data['train_dir']
train_txt = my_data['train_txt']
val_dir = my_data['val_dir']
val_txt = my_data['val_txt']


def get_files_list(dir):
    files_list = []
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            curr_file = parent.split(os.sep)[-1]
            if curr_file == my_labels[0]:
                labels = 0
            elif curr_file == my_labels[1]:
                labels = 1
            elif curr_file == my_labels[2]:
                labels = 2
            elif curr_file == my_labels[3]:
                labels = 3
            elif curr_file == my_labels[4]:
                labels = 4
            files_list.append([os.path.join(curr_file, filename), labels])
    return files_list


def write_txt(content, filename, mode='w'):
    with open(filename, mode) as f:
        for line in content:
            str_line = ""
            for col, data in enumerate(line):
                if not col == len(line) - 1:
                    str_line = str_line + str(data) + " "
                else:
                    str_line = str_line + str(data) + "\n"
            f.write(str_line)


if __name__ == '__main__':
    train_data = get_files_list(train_dir)
    write_txt(train_data, train_txt, mode='w')

    val_data = get_files_list(val_dir)
    write_txt(val_data, val_txt, mode='w')
