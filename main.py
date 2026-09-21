import cv2


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CAMINHO_MASCARA = "image/palhaco.png"
INTENSIDADE_COLORIZE = 0.70


# ============================================================
# CARREGAMENTO DA MÁSCARA
# ============================================================

def carregar_mascara(caminho):

    imagem = cv2.imread(caminho, cv2.IMREAD_UNCHANGED)

    if imagem is None:
        print(f"Erro ao carregar a máscara: {caminho}")
        return None

    if len(imagem.shape) != 3 or imagem.shape[2] != 4:
        print("A máscara precisa ser uma PNG com fundo transparente.")
        return None

    return imagem


# ============================================================
# APLICAÇÃO DA MÁSCARA SOBRE O ROSTO
# ============================================================

def aplicar_mascara(
    frame,
    mascara,
    x,
    y,
    largura,
    altura,
    escala=1.35,
    deslocamento_x=0.0,
    deslocamento_y=0.0
):

    if mascara is None:
        return

    altura_frame, largura_frame = frame.shape[:2]

    largura_mascara = int(largura * escala)
    altura_mascara = int(altura * escala)

    if largura_mascara <= 0 or altura_mascara <= 0:
        return

    centro_x = x + largura // 2
    centro_y = y + altura // 2

    novo_x = centro_x - largura_mascara // 2
    novo_y = centro_y - altura_mascara // 2

    novo_x += int(largura * deslocamento_x)
    novo_y += int(altura * deslocamento_y)

    mascara_redimensionada = cv2.resize(
        mascara,
        (largura_mascara, altura_mascara),
        interpolation=cv2.INTER_AREA
    )

    x1 = max(0, novo_x)
    y1 = max(0, novo_y)

    x2 = min(
        largura_frame,
        novo_x + largura_mascara
    )

    y2 = min(
        altura_frame,
        novo_y + altura_mascara
    )

    if x1 >= x2 or y1 >= y2:
        return

    mascara_x1 = x1 - novo_x
    mascara_y1 = y1 - novo_y

    mascara_x2 = mascara_x1 + (x2 - x1)
    mascara_y2 = mascara_y1 + (y2 - y1)

    mascara_recortada = mascara_redimensionada[
        mascara_y1:mascara_y2,
        mascara_x1:mascara_x2
    ]

    mascara_rgb = mascara_recortada[:, :, :3]

    alpha = (
        mascara_recortada[:, :, 3].astype(float)
        / 255.0
    )

    alpha = alpha[:, :, None]

    regiao = frame[
        y1:y2,
        x1:x2
    ]

    resultado = (
        alpha * mascara_rgb
        +
        (1.0 - alpha) * regiao
    )

    frame[
        y1:y2,
        x1:x2
    ] = resultado.astype("uint8")


# ============================================================
# FILTRO COLORIZE
# ============================================================

def filtro_colorize(frame, intensidade=0.70):

    cinza = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    colorizado = cv2.applyColorMap(
        cinza,
        cv2.COLORMAP_JET
    )

    resultado = cv2.addWeighted(
        frame,
        1.0 - intensidade,
        colorizado,
        intensidade,
        0
    )

    return resultado


# ============================================================
# FILTRO GRAYSCALE
# ============================================================

def filtro_grayscale(frame):

    cinza = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    cinza_bgr = cv2.cvtColor(
        cinza,
        cv2.COLOR_GRAY2BGR
    )

    return cinza_bgr


# ============================================================
# CARREGAMENTO DO CLASSIFICADOR HAAR CASCADE
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Erro ao carregar o Haar Cascade.")
    exit()


# ============================================================
# CARREGAMENTO DO FILTRO DE MÁSCARA
# ============================================================

mascara = carregar_mascara(
    CAMINHO_MASCARA
)


# ============================================================
# INICIALIZAÇÃO DA WEBCAM
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro ao acessar a webcam.")
    exit()


# ============================================================
# DEFINIÇÃO DO MODO INICIAL
# ============================================================

modo_filtro = 0


# ============================================================
# LOOP PRINCIPAL
# ============================================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("Erro ao capturar imagem da câmera.")
        break


    # ========================================================
    # ESPELHAMENTO DA IMAGEM
    # ========================================================

    frame = cv2.flip(frame, 1)


    # ========================================================
    # PREPARAÇÃO DA IMAGEM PARA DETECÇÃO
    # ========================================================

    cinza_deteccao = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # DETECÇÃO DE ROSTOS
    # ========================================================

    rostos = face_cascade.detectMultiScale(
        cinza_deteccao,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )


    # ========================================================
    # APLICAÇÃO DO FILTRO COLORIZE
    # ========================================================

    if modo_filtro == 2:

        frame = filtro_colorize(
            frame,
            INTENSIDADE_COLORIZE
        )


    # ========================================================
    # APLICAÇÃO DO FILTRO GRAYSCALE
    # ========================================================

    elif modo_filtro == 3:

        frame = filtro_grayscale(frame)


    # ========================================================
    # PROCESSAMENTO DOS ROSTOS DETECTADOS
    # ========================================================

    for (x, y, largura, altura) in rostos:


        # ====================================================
        # APLICAÇÃO DA MÁSCARA
        # ====================================================

        if modo_filtro == 1:

            aplicar_mascara(
                frame,
                mascara,
                x,
                y,
                largura,
                altura,
                escala=1.35,
                deslocamento_x=0.0,
                deslocamento_y=-0.05
            )


        # ====================================================
        # DESENHO DO RETÂNGULO DE DETECÇÃO
        # ====================================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + largura, y + altura),
            (0, 255, 0),
            2
        )


        # ====================================================
        # EXIBIÇÃO DO TEXTO DE DETECÇÃO
        # ====================================================

        cv2.putText(
            frame,
            "Rosto Detectado",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


    # ========================================================
    # IDENTIFICAÇÃO DO MODO ATUAL
    # ========================================================

    if modo_filtro == 0:
        nome_modo = "Sem filtro"

    elif modo_filtro == 1:
        nome_modo = "Mascara"

    elif modo_filtro == 2:
        nome_modo = "Colorize"

    else:
        nome_modo = "Grayscale"


    # ========================================================
    # EXIBIÇÃO DO MENU DE CONTROLES
    # ========================================================

    cv2.putText(
        frame,
        "0 - Sem filtro",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "1 - Mascara",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "2 - Colorize",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "3 - Grayscale",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Q - Sair",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # EXIBIÇÃO DO MODO ATUAL
    # ========================================================

    cv2.putText(
        frame,
        f"Modo atual: {nome_modo}",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    # ========================================================
    # EXIBIÇÃO DA QUANTIDADE DE ROSTOS
    # ========================================================

    cv2.putText(
        frame,
        f"Rostos: {len(rostos)}",
        (20, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 255),
        2
    )


    # ========================================================
    # EXIBIÇÃO DA JANELA
    # ========================================================

    cv2.imshow(
        "Face Tracking - OpenCV",
        frame
    )


    # ========================================================
    # LEITURA DO TECLADO
    # ========================================================

    tecla = cv2.waitKey(1) & 0xFF


    # ========================================================
    # ENCERRAMENTO DO PROGRAMA
    # ========================================================

    if tecla == ord("q"):
        break


    # ========================================================
    # SELEÇÃO DO MODO SEM FILTRO
    # ========================================================

    elif tecla == ord("0"):

        modo_filtro = 0
        print("Modo selecionado: Sem filtro")


    # ========================================================
    # SELEÇÃO DO FILTRO DE MÁSCARA
    # ========================================================

    elif tecla == ord("1"):

        if mascara is not None:

            modo_filtro = 1
            print("Modo selecionado: Máscara")

        else:

            print("Não foi possível ativar a máscara.")


    # ========================================================
    # SELEÇÃO DO FILTRO COLORIZE
    # ========================================================

    elif tecla == ord("2"):

        modo_filtro = 2
        print("Modo selecionado: Colorize")


    # ========================================================
    # SELEÇÃO DO FILTRO GRAYSCALE
    # ========================================================

    elif tecla == ord("3"):

        modo_filtro = 3
        print("Modo selecionado: Grayscale")


# ============================================================
# LIBERAÇÃO DOS RECURSOS
# ============================================================

camera.release()
cv2.destroyAllWindows()