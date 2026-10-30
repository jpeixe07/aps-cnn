# APS — Redes Convolucionais

O projeto treina uma CNN própria em MNIST e CIFAR-10, compara
com ResNet-18 pré-treinada no ImageNet e mede o efeito de desligar o dropout.

## Ambiente

Requer Python 3.10+ e PyTorch com torchvision. Na raiz do projeto:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Os datasets são baixados automaticamente para `data/` na primeira execução.
O ResNet-18 baixa os pesos oficiais do ImageNet pelo torchvision.

## Execuções

Treino principal (CNN com dropout), em cada conjunto:

```bash
python -m aps_cnn.train --dataset mnist --mode baseline --epochs 10
python -m aps_cnn.train --dataset cifar10 --mode baseline --epochs 10
```

Comparação com modelo pré-treinado:

```bash
python -m aps_cnn.train --dataset mnist --mode pretrained --epochs 6 --batch-size 128
python -m aps_cnn.train --dataset cifar10 --mode pretrained --epochs 6 --batch-size 128
```

Estudo de ablação já preparado (mesmas CNN, divisão, semente e métricas; dropout
desligado):

```bash
python -m aps_cnn.train --dataset mnist --mode ablation --epochs 10
python -m aps_cnn.train --dataset cifar10 --mode ablation --epochs 10
```

Para uma execução curta que verifica o fluxo sem treinar o dataset inteiro, use,
por exemplo, `--limit 512 --epochs 1`. Remova `--limit` para os resultados finais.
Mantenha os mesmos `--seed`, quantidade de épocas, batch size e dataset ao comparar
baseline e ablação. A avaliação no conjunto de teste acontece apenas ao fim; a
melhor época é selecionada pelo conjunto de validação.

Uma execução pode ser estendida com `--resume --epochs N`; isso continua do melhor
checkpoint de pesos e preserva o histórico, mas reinicia o estado do otimizador.

## Artefatos e métricas

Cada execução grava em `runs/<dataset>/<mode>/`:

- `best_model.pt`: pesos da época com melhor acurácia de validação;
- `metrics.json`: parâmetros, histórico, duração, loss e métricas no teste;
- `training.png`: curvas de loss e acurácia de treino e validação.

Depois de executar os modos desejados, gere uma tabela e gráficos comparativos com:

```bash
python -m aps_cnn.summarize
```

O resumo fica em `results/metrics_summary.csv`, `results/comparison.png`,
`results/ablation.png`, `results/ablation_summary.csv` e `results/ablation_effect.png`.
O resumo verifica automaticamente se baseline e ablação usam a mesma seed,
épocas, batch e quantidade de exemplos de teste. Essa pasta também recebe cópias
das métricas detalhadas e curvas de cada execução para que os resultados acompanhem
o código no repositório.

No teste são calculados acurácia, precisão, recall e F1 macro, além da matriz de
confusão. Esses mesmos indicadores são salvos para comparar a CNN própria, a
ResNet-18 e, depois, as versões com e sem dropout. Use um diretório de saída limpo
por execução se quiser guardar vários seeds; no uso padrão, repetir o mesmo modo
substitui seus artefatos anteriores.

## Decisões do experimento

- CNN própria: três blocos convolucionais (32/64/128 canais), BatchNorm e duas camadas de max-pooling. Em seguida, AdaptiveAvgPool2d((4, 4)) reduz os mapas de características para 4×4, que são enviados ao classificador MLP.
- Regularização da CNN: Dropout2d nas primeiras convoluções e Dropout no
  classificador, ambos removidos em conjunto na ablação.
- Referência pré-treinada: ResNet-18 com pesos ImageNet congelados e cabeça de
  classificação nova e treinável (linear probe). MNIST é convertido em três
  canais; os dois conjuntos são redimensionados a 64×64 para a extração de
  características em CPU.
- Reprodutibilidade: seed padrão 42, divisão treino/validação obtida por uma
  ordem embaralhada de forma determinística, AdamW e Cross Entropy. O carregamento
  usa `--workers 0` por padrão para funcionar em ambientes que bloqueiam subprocessos.

O tempo de execução depende do hardware; a opção pré-treinada pode exigir mais
memória e baixar pesos na primeira execução. Resultados definitivos devem vir das
execuções completas, sem `--limit`.
