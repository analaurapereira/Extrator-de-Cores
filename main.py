import cv2
import numpy as np

# Dicionário de cores com limites [Mínimo, Máximo] no formato HSV
cores_referencia = {
    "Preto":    [np.array([0, 0, 0]), np.array([180, 255, 50])],
    "Branco":   [np.array([0, 0, 200]), np.array([180, 50, 255])],
    "Vermelho": [np.array([0, 120, 70]), np.array([10, 255, 255])],
    "Amarelo":  [np.array([20, 100, 100]), np.array([30, 255, 255])],
    "Verde":    [np.array([36, 100, 40]), np.array([86, 255, 255])],
    "Azul":     [np.array([94, 80, 20]),  np.array([126, 255, 255])],
    "Roxo":     [np.array([130, 50, 50]), np.array([160, 255, 255])],
    "Rosa":     [np.array([160, 50, 50]), np.array([179, 255, 255])]
}

img = cv2.imread('teste.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
total_pixels = img.shape[0] * img.shape[1]

resultados = {}

for nome, limites in cores_referencia.items():
    # Cria a máscara para a cor específica
    mask = cv2.inRange(hsv, limites[0], limites[1])
    
    # Conta os pixels dentro do intervalo
    contagem = cv2.countNonZero(mask)
    proporcao = (contagem / total_pixels) * 100
    
    resultados[nome] = {
        "proporcao": f"{proporcao:.2f}%",
        "valor_real": proporcao
    }

# Ordena do maior para o menor com base no valor da proporção
resultados_ordenados = sorted(resultados.items(), key=lambda item: item[1]['valor_real'], reverse=True)

for cor, dados in resultados_ordenados:
    print(f"Cor: {cor} | Proporção: {dados['proporcao']}")