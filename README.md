# 🧠 Sistema de Diagnóstico Tumoral

## 📖 Descrição do Projeto
Este projeto é um sistema simulado de diagnóstico clínico que utiliza técnicas de Visão Computacional para analisar imagens de Ressonância Magnética (MRI) do cérebro. O script interage com o usuário para coletar dados básicos do paciente (sexo, peso, altura) e, em seguida, avalia a imagem de MRI para detectar, segmentar e destacar anomalias (como possíveis massas tumorais). O objetivo final é não apenas localizar a patologia, mas também estimar a gravidade ao calcular a porcentagem da área comprometida do cérebro e seu peso aproximado. 

## 💻 Tecnologias Utilizadas
- **Python 3**
- **OpenCV (`cv2`)**
- **NumPy (`numpy`)**
- **KaggleHub (`kagglehub`)**
- **OS (`os`)**
- **Random (`random`)**

## 🚀 Como Rodar o Projeto na Própria Máquina
Siga o passo a passo abaixo para executar o sistema na sua máquina local:

1. **Clone o Repositório:** Abra o seu Console (ou Terminal) e baixe o projeto:
   ```bash
   git clone https://github.com/analaurapereira/Sistema-de-Diagnostico-Tumoral.git
   ```
2. **Navegue até o Projeto:** Entre na pasta gerada pelo clone, onde o arquivo `main.py` está salvo (exemplo: `cd Extrator_de_cores`).
3. **Crie um Ambiente Virtual (*Recomendado*):**
   Isso isola as bibliotecas do projeto da sua máquina. No terminal, digite:
   ```bash
   python3 -m venv venv
   ```
   Ative o ambiente *(No Linux/Mac: `source venv/bin/activate` | No Windows: `venv\Scripts\activate`)*.
4. **Instale as Dependências (Bibliotecas):**
   Com o terminal pronto, execute o comando para baixar as ferramentas de visão computacional:
   ```bash
   pip install opencv-python numpy kagglehub
   ```
   *(Nota: As bibliotecas `os` e `random` já são embutidas de fábrica no Python e não precisam de instalação).*
5. **Execute o Sistema!**
   Para rodar o programa, execute o comando:
   ```bash
   python main.py
   ```
   O script começará a interagir com você no próprio terminal.


## ⚙️ Como o Projeto Funciona
Ao executar o arquivo principal (`main.py`), a seguinte sequência acontece:

1. **Coleta de Dados**: O sistema solicita via terminal informações biológicas do paciente (sexo, peso, altura).
2. **Download Dinâmico do Dataset**: Se a base não estiver salva no computador, o script faz o download pelo Kaggle separando as ressonâncias entre *yes* (com tumor) e *no* (saudáveis).
3. **Sorteio Aleatório**: Tendo os arquivos mapeados em disco, ele simula um paciente sorteando uma das ressonâncias aleatoriamente.
4. **Análises e Processamento Analítico**: O script trabalha a foto convertendo-a para identificar todo o espaço preenchido pelo cérebro e, posteriormente, escaneia este mapeamento focando em pontos muito iluminados atípicos (anomalias).
5. **Cálculos Matemáticos-Clínicos**: Tendo mapeado através de pixels a massa craniana inteira em relação à massa tumoral da imagem, é feito o cálculo do grau do dano. Com base na altura e sexo informados e da estimativa, o peso total do cérebro em gramas é calculado, derivando-se assim o peso que essa anomalia teria.
6. **Resultados**: O script emite alertas via terminal e aciona o pop-up das imagem geradas em 4 camadas que mostram tudo o que o código analisou na radiologia (onde ele aplica a linha marcando a anomalia em vermelho, por exemplo).

<--Adicionar minhas imagens e vídeos-->

<br>

## 🤖 Técnicas de IA e Visão Computacional Envolvidas

Para que o sistema consiga analisar a imagem de ressonância e encontrar uma anomalia, ele segue alguns passos para "limpar" e organizar o que ele está vendo. A abordagem utiliza passos visuais progressivos:

- **1. Simplificando as Cores (Escala de Cinza):** 
Em vez de tentar processar diferentes tons de cor, o script transforma a imagem inteira em preto e branco. Isso é útil porque, em exames de imagem, o detalhe mais importante é o brilho da região.

- **2. Desfoque/Blur:**
Muitas imagens médicas vêm com ruídos da máquina do hospital. Para evitar que o sistema confunda esses pontos com um tumor, aplicamos um embaçamento que esfumaça pixels distorcidos, limpando a tela para um diagnóstico seguro.

- **3. Separando o Claro do Escuro:**
Aqui o sistema usa uma tesoura nas cores: "Tudo que for escuro vira preto absoluto, e o que for mais claro vira branco absoluto". O script faz dois cortes:
  - *Corte Suave:* Remove o fundo escuro do exame inteiro para focar só na massa encefálica.
  - *Corte Agressivo:* Tendo apenas o cérebro em vista, ele foca estritamente para que brilhem apenas as partes super expostas e muito mais brancas que o normal (onde se formam as anomalias e tumores).

- **4. Preenchimento de falhas:**
Nesse estágio, a massa tumoral já foi separada, mas ela pode ter saído com falhas, como se estivesse quebrada. O sistema faz um processo para fechar os buracos no meio do tumor, unindo-o por completo; e outro para varrer essas migalhas de poeira branca que sobraram, evitando falsos alarmes de "mini-tumores" onde não tem nada.

- **5. Desenhando Bordas:**
Com a área do tumor totalmente isolada, o sistema desenha uma linha exata contornando as bordas dessa massa branca. Com esse contorno, é possível usar fórmulas matemáticas na área para medir seu tamanho pixel por pixel. Se for uma mancha expressiva, ele emite o Alerta, e o risco fica delimitado na tela.

## 🧮 Cálculos Utilizados

O script realiza estimativas matemáticas para fornecer um prognóstico básico. Os métodos empregados são:

- **Porcentagem de Área Afetada:**
A divisão da área detectada da anomalia (em pixels) pela área total detectada para a massa craniana inteira (em pixels). O resultado é multiplicado por 100 para entregar o valor em porcentagem (`% da anomalia = (área suspeita / área do cérebro) * 100`).

- **Estimativa de Peso do Cérebro em Gramas:**
Utiliza a fórmula baseada em métodos antropométricos, que correlaciona a altura do indivíduo (em cm) e o sexo biológico para apontar uma estimativa do peso do cérebro do paciente.
  - Para Homens (biológico): `920 + 2.70 * Altura_em_cm`
  - Para Mulheres (biológico): `748 + 3.10 * Altura_em_cm`

- **Estimativa de Peso de Massa Tumoral:**
Aplica uma proporção direta utilizando a porcentagem extraída visando aplicar sobre a estimativa teórica do peso do cérebro do paciente (`Peso do tumor = Peso total do cérebro * (% da anomalia) / 100`).

<br>

## 🔮 Expansão com Deep Learning (IA)

Atualmente, o projeto emprega técnicas de **Visão Computacional Tradicional** usando apenas OpenCV. Esse método está vulnerável a *falsos positivos* causados por brilhos indevidos ou ruídos intensos na máquina de ressonância. Para ampliar e refinar a acurácia para uso clínico real, o desenvolvimento pode integrar **Redes Neurais Artificiais** e *Deep Learning*:

- **Arquiteturas como *U-Net*:** Em vez de deduzir massas pelas cores claras generalizadas, um modelo *U-Net* treinado pode identificar com exatidão onde o limite de um tumor real termina e onde o cérebro saudável começa de forma não-linear.
- **Redes Convolucionais para Classificação:** Além da simples detecção, esses modelos poderiam classificar os tumores em padrões conhecidos da literatura neurológica (Meningioma, Glioma, Tumor Pituitário, etc.) ou se é maligno/benigno.
- **Ferramentas e Bibliotecas que podem ser aplicadas no futuro:**
  - `TensorFlow` e `PyTorch`: Para a criação estrutural, treino e otimização dos modelos.
  - `Keras`: Para auxiliar na montagem e visualização das camadas de nós da Rede Neural.
  - `Scikit-Learn`: Para a mensuração, tabelas de confusão e métricas de validação clínica dos resultados da IA.