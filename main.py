import os
os.environ['QT_LOGGING_RULES'] = '*=false'

import cv2
import numpy as np
import kagglehub
import random

print("=== SISTEMA DE DIAGNÓSTICO ===")
sexo = input("Digite o sexo biológico do paciente (M/F): ")
peso = input("Digite o peso do paciente (kg): ")
altura = input("Digite a altura do paciente (cm): ")

# Baixa a base de dados de imagens de ressonância (Kaggle)
path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")

# Escolhe aleatoriamente uma imagem com ou sem tumor
pasta_escolhida = random.choice(['yes', 'no'])
subpasta = os.path.join(path, pasta_escolhida)

img_name = random.choice([f for f in os.listdir(subpasta) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
img_path = os.path.join(subpasta, img_name)

# Lê a imagem e cria uma cópia para desenhar o resultado
img = cv2.imread(img_path)
diagnostico = img.copy()

# Converte a imagem BGR (colorida) para Cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplica um desfoque suave para remover pequenos artefatos
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 1. ENCONTRAR O CÉREBRO COMPLETO
# Aplica limite baixo de cor para enxergar apenas a cabeça toda
_, thresh_cerebro = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)

# Limpa sujeiras morfológicas da imagem
thresh_cerebro = cv2.morphologyEx(thresh_cerebro, cv2.MORPH_OPEN, kernel, iterations=1)

# Traça o contorno do cérebro inteiro
contours_cerebro, _ = cv2.findContours(thresh_cerebro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

area_cerebro = 0
if contours_cerebro:
    c_cerebro_max = max(contours_cerebro, key=cv2.contourArea)
    area_cerebro = cv2.contourArea(c_cerebro_max)

# 2. ENCONTRAR O TUMOR (ÁREAS MAIS CLARAS)
# Aplica um limite alto para enxergar só partes muito brancas (anomalias)
_, thresh = cv2.threshold(blur, 155, 255, cv2.THRESH_BINARY)

# Preenche buracos e limpa pontos fora da anomalia principal
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Extrai o contorno da massa tumoral
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

tumor_encontrado = False
area_suspeita = 0

if contours:
    # Seleciona o maior contorno branco achado
    c_max = max(contours, key=cv2.contourArea)
    area_suspeita = cv2.contourArea(c_max)
    
    # Verifica se a área suspeita é maior que 300 pixels
    if area_suspeita > 300:
        tumor_encontrado = True
        
        # Desenha a linha vermelha em volta da área suspeita
        cv2.drawContours(diagnostico, [c_max], -1, (0, 0, 255), 2)
        

# --- ESTIMATIVA DE PESO DO CÉREBRO ---
# Utilizando a fórmula baseada em altura e sexo (Ho et al.)
peso_cerebro_g = 0
try:
    alt_val = float(altura)
    if sexo.upper().startswith('M'):
        peso_cerebro_g = 920 + 2.70 * alt_val
    elif sexo.upper().startswith('F'):
        peso_cerebro_g = 748 + 3.10 * alt_val
    else:
        peso_cerebro_g = 834 + 2.90 * alt_val  # Média
except ValueError:
    pass

# --- IMPRIMIR DADOS DO EXAME ---
print("--- Relatório de Scanner MRI (Cérebro) ---")
print(f"Paciente: Sexo {sexo.upper()} | Peso: {peso}kg | Altura: {altura}cm")
print(f"Arquivo analisado: {img_name}")
print(f"Diagnóstico real do dataset: {pasta_escolhida.upper()}")

if tumor_encontrado:
    print(f"\n[ALERTA]: Possível massa tumoral detectada pelo algoritmo!")
    if area_cerebro > 0:
        pct = (area_suspeita / area_cerebro) * 100
        print(f">> Porcentagem da massa afetada pela anomalia: {pct:.2f}%")
        
        if peso_cerebro_g > 0:
            peso_tumor_g = peso_cerebro_g * (pct / 100)
            print(f">> Peso estimado do cérebro: {peso_cerebro_g:.2f}g")
            print(f">> Peso estimado da área afetada: {peso_tumor_g:.2f}g")
else:
    print("\n[NORMAL]: Nenhuma anomalia significativa foi detectada nos testes primários.")

# Abre na tela as 4 fases do processamento
cv2.imshow('MRI Original', img)
cv2.imshow('Massa Craniana', thresh_cerebro)
cv2.imshow('Processamento(Threshold)', thresh)
cv2.imshow('Diagnostico', diagnostico)

print("\n(Pressione qualquer tecla nas janelas de imagem para fechar o programa)")
cv2.waitKey(0)
cv2.destroyAllWindows()
