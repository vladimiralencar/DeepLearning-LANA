########################################################################
#
# Download e extração de dados
#
# Implementado em Python 3.6
#
########################################################################

import sys
import os
import urllib.request
import tarfile
import zipfile

########################################################################


def _print_download_progress(count, block_size, total_size):

    pct_complete = float(count * block_size) / total_size

    msg = "\r- Download em andamento: {0:.1%}".format(pct_complete)

    # Print 
    sys.stdout.write(msg)
    sys.stdout.flush()


########################################################################


def maybe_download_and_extract(url, download_dir):

    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

   
    if not os.path.exists(file_path):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        file_path, _ = urllib.request.urlretrieve(url = url, filename = file_path, reporthook = _print_download_progress)

        print()
        print("Download concluído. Extraindo os arquivos.")

        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Feito.")
    else:
        print("Os dados aparentemente já foram baixados e descompactados.")


########################################################################
