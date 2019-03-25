########################################################################
#
# Classe para criar o dataset com todos os arquivos no diretório
#
#
# Implementado em Python 3.6
#
########################################################################

import numpy as np
import os
from cache import cache

########################################################################


def one_hot_encoded(class_numbers, num_classes = None):
   
    if num_classes is None:
        num_classes = np.max(class_numbers) - 1

    return np.eye(num_classes, dtype = float)[class_numbers]


########################################################################


class DataSet:
    def __init__(self, in_dir, exts = '.jpg'):
        in_dir = os.path.abspath(in_dir)

        self.in_dir = in_dir

        self.exts = tuple(ext.lower() for ext in exts)

        self.class_names = []

        self.filenames = []

        self.filenames_test = []

        self.class_numbers = []

        self.class_numbers_test = []

        self.num_classes = 0

        for name in os.listdir(in_dir):
            current_dir = os.path.join(in_dir, name)

            if os.path.isdir(current_dir):
                self.class_names.append(name)

                filenames = self._get_filenames(current_dir)

                self.filenames.extend(filenames)

                class_number = self.num_classes

                class_numbers = [class_number] * len(filenames)

                self.class_numbers.extend(class_numbers)

                filenames_test = self._get_filenames(os.path.join(current_dir, 'test'))

                self.filenames_test.extend(filenames_test)

                class_numbers = [class_number] * len(filenames_test)

                self.class_numbers_test.extend(class_numbers)

                self.num_classes += 1

    
    def _get_filenames(self, dir):

        filenames = []

        if os.path.exists(dir):
            for filename in os.listdir(dir):
                if filename.lower().endswith(self.exts):
                    filenames.append(filename)

        return filenames


    def get_paths(self, test=False):

        if test:
            filenames = self.filenames_test
            class_numbers = self.class_numbers_test

            test_dir = "test/"
        else:
            filenames = self.filenames
            class_numbers = self.class_numbers

            test_dir = ""

        for filename, cls in zip(filenames, class_numbers):
            path = os.path.join(self.in_dir, self.class_names[cls], test_dir, filename)

            yield path


    def get_training_set(self):

        return list(self.get_paths()), np.asarray(self.class_numbers), one_hot_encoded(class_numbers = self.class_numbers, num_classes = self.num_classes)


    def get_test_set(self):

        return list(self.get_paths(test = True)), np.asarray(self.class_numbers_test), one_hot_encoded(class_numbers = self.class_numbers_test, num_classes = self.num_classes)


########################################################################


def load_cached(cache_path, in_dir):

    print("Criando conjuntos de dados a partir dos arquivos em: " + in_dir)

    dataset = cache(cache_path = cache_path, fn = DataSet, in_dir = in_dir)

    return dataset


########################################################################
