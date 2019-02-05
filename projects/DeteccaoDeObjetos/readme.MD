O Arquivo está dividido em vários pedaços. 
Fazer o download do arquivo .tar.gz e executar o passos:

1. Juntar novamente o arquivo
cat DeteccaoDeImagens-s* > DeteccaoDeImagens-Modelo_SSD-FULL.tar.gz

2. Descompactar
tar xvzf DeteccaoDeImagens-Modelo_SSD-FULL.tar.gz

------------------------------------------------------------------------------------------------------

O arquivo foi criado como os seguintes comandos:

1. Compactar um diretório/folder
tar -czvf directorios.tar.gz folder
tar -czvf DeteccaoDeImagens-Modelo_SSD.tar.gz DeteccaoDeImagens-Modelo_SSD

2. Separar o arquivo em pedaços de 20MB
split -b 20000000 DeteccaoDeImagens-Modelo_SSD.tar.gz DeteccaoDeImagens-split-

