# Projeto de Aprendizado de Máquina (Rede Neural)

Este projeto utiliza a biblioteca PyTorch para criar uma rede neural que é treinada no conjunto de dados MNIST.

## Dependências

O projeto depende das seguintes bibliotecas Python:

- numpy
- torch
- torchvision
- matplotlib

## Código

O código começa importando as bibliotecas necessárias:

```python
import numpy as np
import torch
import torch.nn.functional as f
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
```

Em seguida, definimos a transformação que queremos aplicar às nossas imagens. Neste caso, estamos convertendo as imagens em tensores usando transforms.ToTensor():

```
transform = transforms.ToTensor()
```

Carregamos o conjunto de dados de treinamento e validação do MNIST. O conjunto de dados é baixado se ainda não estiver presente:

```
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

valset= datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
```

A seguir, pegamos um lote de imagens e etiquetas do carregador de dados de treinamento:

```
dataiter = iter(trainloader)
imagens, etiquetas = next(dataiter)
```

Visualizamos a primeira imagem do lote:

```
plt.imshow(imagens[0].numpy().squeeze(), cmap='gray_r')
```

Finalmente, imprimimos as dimensões do tensor de cada imagem e etiqueta para verificar:

```
print(imagens[0].shape) #Para verificar as dimensões do Tensor em cada iimagem.
print(etiquetas[0].shape) #para verificar a dimensões do Tensor de cada etiqueta.
```

Este projeto está hospedado no Google Colab.

Espero que isso ajude! Se você tiver mais perguntas ou precisar de mais detalhes, sinta-se à vontade para perguntar.

Como Executar
```
Clone este repositório: git clone https://seu-repositorio.git
Instale as dependências: pip install -r requirements.txt (se houver)
Execute o código: python main.py
```
Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues, propor melhorias ou enviar pull requests.

Licença
Este projeto é licenciado sob a MIT License.

Agradecimentos
Agradecemos ao PyTorch e aos desenvolvedores de bibliotecas relacionadas.

🌐 Fontes:

(udacity/deep-learning-v2-pytorch)

(bharathgs/Awesome-pytorch-list)

(ultralytics/mnist)

(mrdbourke/pytorch-deep-learning)

(pytorch/examples)

(chandpes/ML-DL-Compendium)

