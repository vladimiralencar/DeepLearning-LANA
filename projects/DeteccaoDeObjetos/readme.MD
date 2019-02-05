O Arquivo está divido em vários pedaços. Fazer o download do arquivo .tar.gz e executar o passos:

1. Junta novamente o arquivo
cat DeteccaoDeImagens-s* > DeteccaoDeImagens-Modelo_SSD-FULL.tar.gz

2. descompacta
tar xvzf DeteccaoDeImagens-Modelo_SSD-FULL.tar.gz

------------------------------------------------------------------------------------------------------
O arquivo foi criado como os comandos:

1. Compacta um diretório/folder
tar -czvf directorios.tar.gz folder
tar -czvf DeteccaoDeImagens-Modelo_SSD.tar.gz DeteccaoDeImagens-Modelo_SSD

2. separa o arquivo em pedaços de 20MB
split -b 20000000 DeteccaoDeImagens-Modelo_SSD.tar.gz DeteccaoDeImagens-split-

