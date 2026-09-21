# Face Tracking com Filtros em OpenCV

Projeto desenvolvido em Python utilizando a biblioteca **OpenCV** para realizar detecção facial em tempo real através da webcam.

A aplicação utiliza um classificador **Haar Cascade** para detectar rostos e permite alternar entre diferentes modos de visualização e filtros.

## Funcionalidades

O programa possui quatro modos:

* **Sem filtro:** exibe a imagem original da webcam.
* **Máscara:** aplica uma imagem PNG transparente sobre o rosto detectado.
* **Colorize:** aplica um efeito de coloração utilizando `COLORMAP_JET`.
* **Grayscale:** transforma a imagem da webcam em tons de cinza.

Além dos filtros, o programa:

* Detecta rostos em tempo real.
* Exibe um retângulo ao redor de cada rosto detectado.
* Mostra o texto `Rosto Detectado`.
* Exibe a quantidade de rostos encontrados.
* Permite detectar mais de uma pessoa simultaneamente.
* Permite alternar os filtros utilizando o teclado.

## Tecnologias utilizadas

* Python
* OpenCV
* Haar Cascade
* Processamento de imagens em tempo real

## Estrutura do projeto

```text
face_tracking-opencv/
│
├── main.py
├── README.md
│
└── image/
    └── palhaco.png
```

O arquivo `mascara.png` deve possuir fundo transparente para que a sobreposição sobre o rosto funcione corretamente.

## Instalação

### 1. Instalar o Python

É necessário possuir o Python instalado no computador.

Para verificar a instalação:

```bash
python --version
```

Dependendo do sistema operacional, também pode ser necessário utilizar:

```bash
python3 --version
```

### 2. Instalar o OpenCV

No terminal, execute:

```bash
pip install opencv-python
```

Caso esteja utilizando `pip3`:

```bash
pip3 install opencv-python
```

## Executando o projeto

Abra o terminal dentro da pasta do projeto e execute:

```bash
python main.py
```

Ou:

```bash
python3 main.py
```

A webcam será aberta e o programa começará a detectar rostos.

## Controles

Durante a execução, utilize as seguintes teclas:

| Tecla | Função              |
| ----- | ------------------- |
| `0`   | Sem filtro          |
| `1`   | Máscara             |
| `2`   | Colorize            |
| `3`   | Grayscale           |
| `Q`   | Encerrar o programa |

## Detecção facial

A detecção dos rostos é realizada utilizando o classificador:

```text
haarcascade_frontalface_default.xml
```

O arquivo faz parte dos classificadores pré-treinados disponibilizados pelo OpenCV.

No código, ele é carregado através de:

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)
```

Para realizar a detecção, cada imagem capturada pela webcam é convertida para tons de cinza.

```python
cinza_deteccao = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
)
```

Em seguida, o Haar Cascade procura regiões da imagem que apresentam características semelhantes às de um rosto.

```python
rostos = face_cascade.detectMultiScale(
    cinza_deteccao,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)
```

Para cada rosto detectado são obtidas quatro informações:

```text
x
y
largura
altura
```

Essas informações representam a posição e o tamanho do rosto dentro da imagem.

## Modo sem filtro

No modo `0`, a imagem original da webcam é exibida normalmente.

Quando um rosto é encontrado, o programa desenha um retângulo verde ao seu redor e apresenta o texto:

```text
Rosto Detectado
```

Esse modo permite visualizar diretamente o funcionamento da detecção facial.

## Filtro de máscara

O modo `1` utiliza uma imagem PNG transparente que é posicionada sobre cada rosto encontrado.

Arquivo utilizado:

```text
filtros/mascara.png
```

A máscara é redimensionada automaticamente de acordo com:

```text
largura do rosto
altura do rosto
```

O programa também utiliza o canal **Alpha** da imagem PNG para controlar sua transparência.

Uma imagem PNG utilizada como máscara possui quatro canais:

```text
B - Blue
G - Green
R - Red
A - Alpha
```

O canal Alpha determina quais partes da imagem serão visíveis ou transparentes.

A máscara acompanha o movimento da pessoa porque sua posição é recalculada em todos os frames capturados pela webcam.

## Filtro Colorize

O modo `2` aplica um efeito de coloração sobre toda a imagem.

Primeiro, o frame é convertido para tons de cinza:

```python
cinza = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
)
```

Depois é aplicado o mapa de cores:

```python
colorizado = cv2.applyColorMap(
    cinza,
    cv2.COLORMAP_JET
)
```

Por fim, a imagem colorizada é misturada com a imagem original:

```python
resultado = cv2.addWeighted(
    frame,
    0.30,
    colorizado,
    0.70,
    0
)
```

O resultado possui um efeito visual semelhante a um mapa térmico.

A intensidade do filtro pode ser alterada através da variável:

```python
INTENSIDADE_COLORIZE = 0.70
```

Valores próximos de `1.0` tornam o efeito mais forte.

Valores próximos de `0.0` deixam a imagem mais próxima da original.

## Filtro Grayscale

O modo `3` transforma a imagem da webcam em tons de cinza.

A conversão é realizada utilizando:

```python
cinza = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
)
```

Depois, a imagem é convertida novamente para três canais:

```python
cinza_bgr = cv2.cvtColor(
    cinza,
    cv2.COLOR_GRAY2BGR
)
```

Essa segunda conversão não devolve as cores da imagem.

Ela apenas permite continuar utilizando elementos coloridos, como o retângulo de detecção e os textos exibidos pelo OpenCV.

## Fluxo de funcionamento

O funcionamento geral da aplicação pode ser representado da seguinte forma:

```text
Webcam
   ↓
Captura do frame
   ↓
Espelhamento da imagem
   ↓
Conversão para escala de cinza
   ↓
Detecção utilizando Haar Cascade
   ↓
Obtenção de X, Y, largura e altura
   ↓
Seleção do filtro
   ↓
Aplicação do efeito
   ↓
Desenho da área de detecção
   ↓
Exibição do resultado
   ↓
Novo frame
```

Esse processo ocorre repetidamente enquanto o programa estiver aberto.

## Detecção e reconhecimento facial

Este projeto realiza **detecção facial**, e não reconhecimento facial.

A detecção responde à pergunta:

```text
Existe um rosto nesta imagem?
```

O programa consegue identificar onde o rosto está localizado, mas não sabe quem é a pessoa.

Já um sistema de reconhecimento facial tentaria responder:

```text
De quem é este rosto?
```

Portanto:

```text
Detecção facial
→ localiza rostos.

Reconhecimento facial
→ tenta identificar a pessoa.
```

## Limitações

O classificador utilizado foi desenvolvido principalmente para detectar rostos vistos de frente.

Por isso, a detecção pode apresentar dificuldades em algumas situações, como:

* Rosto muito distante da câmera.
* Pouca iluminação.
* Rosto parcialmente coberto.
* Uso de máscara facial.
* Uso de óculos escuros.
* Rosto visto completamente de perfil.
* Movimento muito rápido.
* Imagem da webcam com baixa qualidade.

Essas limitações fazem parte da investigação proposta pela atividade.

## Testes sugeridos

O projeto pode ser testado em diferentes situações.

| Teste             | Objetivo                                              |
| ----------------- | ----------------------------------------------------- |
| Distância         | Verificar até onde o rosto continua sendo detectado   |
| Óculos escuros    | Verificar o impacto da região dos olhos               |
| Máscara facial    | Verificar o impacto da região da boca e nariz         |
| Perfil            | Verificar o comportamento com o rosto de lado         |
| Múltiplas pessoas | Verificar a detecção de vários rostos simultaneamente |
| Iluminação        | Comparar ambientes claros e escuros                   |

Os resultados podem variar de acordo com a webcam, iluminação e posição das pessoas.

## Haar Cascade

O Haar Cascade trabalha procurando padrões visuais e diferenças de contraste presentes em imagens de rostos.

Entre as regiões importantes estão:

* Olhos.
* Sobrancelhas.
* Nariz.
* Testa.
* Contorno facial.

O algoritmo foi previamente treinado utilizando exemplos positivos e negativos.

Por isso, neste projeto não é necessário realizar o treinamento de um modelo do zero.

## Referência

Os filtros de imagem utilizados neste projeto foram baseados nos conceitos apresentados pelo OpenCV para aplicação de filtros em vídeo em tempo real:

**OpenCV — Real-time Webcam Filters**

https://opencv.org/blog/opencv-js-real-time-webcam-filters/

O projeto adapta os conceitos apresentados em OpenCV.js para uma implementação utilizando Python e OpenCV.

## Objetivo da atividade

O objetivo do projeto é demonstrar conceitos básicos de visão computacional, incluindo:

* Captura de vídeo em tempo real.
* Processamento de imagens.
* Conversão de espaços de cor.
* Detecção de objetos.
* Haar Cascades.
* Coordenadas de regiões detectadas.
* Aplicação de filtros.
* Sobreposição de imagens transparentes.
* Rastreamento visual baseado em detecção contínua.

## Autor

Projeto desenvolvido para a atividade de **Detecção de Faces e Aplicação de Filtros utilizando OpenCV**.
