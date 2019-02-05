O Arquivo está dividido em vários pedaços.  <br>
Fazer o download do arquivo .tar.gz e executar o passos: <br>

1. Juntar novamente o arquivo <br>
cat DeteccaoDeImagens-s* > DeteccaoDeImagens-Modelo_SSD-FULL.tar.gz <br>

2. Descompactar <br>
tar xvzf DeteccaoDeImagens-Modelo_SSD-FULL.tar.gz <br>

3. Executar o Jupyter Notebook (arquivo .ipynb) <br>

------------------------------------------------------------------------------------------------------

O arquivo foi criado como os seguintes comandos: <br>

1. Compactar um diretório/folder <br>
tar -czvf directorios.tar.gz folder <br>
tar -czvf DeteccaoDeImagens-Modelo_SSD.tar.gz DeteccaoDeImagens-Modelo_SSD <br>

2. Separar o arquivo em pedaços de 20MB <br>
split -b 20000000 DeteccaoDeImagens-Modelo_SSD.tar.gz DeteccaoDeImagens-split- <br>

