<h1 align="center" style="font-weight: bold;">Variational Autoencoder (VAE) & Visualização com Heatmaps 🔍</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white" alt="Keras"/>
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=Matplotlib&logoColor=white" alt="Matplotlib"/>
</p>

<p align="center">
  <a href="#projeto">Projeto</a> •
  <a href="#vae">Sobre o VAE</a> • 
  <a href="#heatmaps">Visualização com Heatmaps</a> 
</p>

---

<h2 id="projeto">📫 Projeto</h2>

Este projeto foi desenvolvido com o objetivo de explorar o comportamento de um **Variational Autoencoder** em tarefas de reconstrução de imagens, e ao mesmo tempo, analisar **como as camadas convolucionais estão reagindo a diferentes entradas** por meio de mapas de ativação (*heatmaps*).

---

<h2 id="vae">🧠 Sobre o VAE (Variational Autoencoder)</h2>

O **VAE** é um tipo de autoencoder probabilístico que aprende uma **distribuição latente contínua** dos dados, permitindo a geração de novas amostras e interpolação no espaço latente.

Diferente dos autoencoders clássicos, o VAE aprende não apenas a reconstruir as imagens, mas também a **distribuição dos dados no espaço latente**, por meio de duas saídas no encoder:
- **Média (μ)**
- **Desvio padrão (σ)**

Esses parâmetros são usados para **amostrar o vetor latente z**, que é passado ao decoder.

Aplicações do VAE incluem:
- Geração de novas imagens
- Compressão de dados
- Aprendizado não supervisionado
- Análise de anomalias

---

<h2 id="heatmaps">🔥 Visualização com Heatmaps</h2>

Para melhor interpretar **como a rede está extraindo características visuais**, são gerados **heatmaps** das ativações das camadas convolucionais do encoder.

Esses mapas mostram:
- Regiões da imagem com **maior ativação**
- Quais partes da entrada estão influenciando mais a codificação
- Insights sobre **atenção espacial da rede**

<p align="center">
  <img src="https://github.com/Lucas-doc26/VAE/blob/main/img.png" alt="Exemplo de heatmap" width="600px">
</p>

A técnica utilizada para gerar os heatmaps é baseada na **extração das ativações** intermediárias da rede, seguida de uma **normalização e sobreposição na imagem original**.
