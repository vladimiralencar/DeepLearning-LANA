########################################################################
#
# Função usada para persistir os dados e carregá-los mais rapidamente
#
# Implementado em Python 3.6
#
########################################################################

import os
import pickle
import numpy as np

########################################################################


def cache(cache_path, fn, *args, **kwargs):

    if os.path.exists(cache_path):
        with open(cache_path, mode='rb') as file:
            obj = pickle.load(file)

        print("- Dados carregados do arquivo de cache: " + cache_path)
    else:

        obj = fn(*args, **kwargs)

        with open(cache_path, mode='wb') as file:
            pickle.dump(obj, file)

        print("- Dados salvos no arquivo de cache: " + cache_path)

    return obj


########################################################################


def convert_numpy2pickle(in_path, out_path):
  
    data = np.load(in_path)

    with open(out_path, mode='wb') as file:
        pickle.dump(data, file)


########################################################################

if __name__ == '__main__':

    def expensive_function(a, b):
        return a * b

    print('Computando expensive_function() ...')

    result = cache(cache_path='cache_expensive_function.pkl', fn=expensive_function, a=123, b=456)

    print('Resultado =', result)

    print()

    class ExpensiveClass:
        def __init__(self, c, d):
            self.c = c
            self.d = d
            self.result = c * d

        def print_result(self):
            print('c =', self.c)
            print('d =', self.d)
            print('result = c * d =', self.result)

    print('Criando objeto ExpensiveClass() ...')

    obj = cache(cache_path='cache_ExpensiveClass.pkl', fn=ExpensiveClass, c=123, d=456)

    obj.print_result()

########################################################################
