import os 
import numpy as np
import pandas as pd 

SEED=42

def cria_dirs():
    dirs = ['PKLot','PUC', 'UFPR04', 'UFPR05', 'CNR', 'Kyoto']
    if not os.path.isdir('CSV'):
        os.makedirs('CSV')
        for dr in dirs:
            os.makedirs(f'CSV/{dr}')

def PKLot(random_state=42):
    cria_dirs()

    def cria_pklot():
        df_final = pd.DataFrame()  
        n_imgs = [102, 102, 102, 103, 103]
        faculdades = ['PUC', 'UFPR04', 'UFPR05']
        tempos = ['Cloudy', 'Rainy', 'Sunny']
        classes = ['Empty', 'Occupied']

        dados = []
        path_pklot = "/opt/datasets/PKLot/PKLotSegmented"
        for faculdade in faculdades:
            for tempo in tempos:
                path_facul_tempo = os.path.join(path_pklot, faculdade, tempo) #"PKLot/PKLotSegmented/PUC/Sunny"
                dias = os.listdir(path_facul_tempo)
                for dia in dias:
                    for classe in classes:
                        path_imgs = os.path.join(path_facul_tempo, dia, classe) #"PKLot/PKLotSegmented/PUC/Sunny/2012-09-12/Empty"
                        
                        if not os.path.isdir(path_imgs):
                            continue

                        imagens = os.listdir(path_imgs)
                        for img in imagens:
                            caminho_img = os.path.join(path_imgs, img)
                            dados.append([faculdade, tempo, dia, caminho_img, classe])

        df = pd.DataFrame(data=dados, columns=['Faculdade', 'Tempo', 'Dia', 'caminho_imagem' ,'classe'])
        df['classe'] = df['classe'].replace({'Empty': 1, 'Occupied': 0})
        df.to_csv("CSV/PKLot/PKLot.csv")

    cria_pklot()

    #Variveis de controle
    df_final = pd.DataFrame()  
    n_imgs = [102, 102, 102, 103, 103]
    faculdades = ['PUC', 'UFPR04', 'UFPR05']
    tempos = ['Cloudy', 'Rainy', 'Sunny']
    classes = ['Empty', 'Occupied']
    df = pd.read_csv('CSV/PKLot/PKLot.csv')

    dias_cada_facul = []
    for faculdade in faculdades:
        dias = df[df["Faculdade"] == f'{faculdade}']["Dia"].unique()
        dias_cada_facul.append(dias)

    # Df de cada uma das faculdades:
    #Treino  
    for i, faculdade in enumerate(faculdades):
        df_facul = df[(df['Faculdade'] == f'{faculdade}')] 

        file_treino = f"CSV/{faculdade}/{faculdade}.csv"
        df_facul_final = df_facul[['caminho_imagem', 'classe']]
        df_facul_final.to_csv(file_treino, index=False)

        primeiros_dias = dias_cada_facul[i][:5]  
        print("Os respectivos dias foram selecionados para treino: ", primeiros_dias)
        dias_cada_facul[i] = dias_cada_facul[i][5:] #Removendo os dias selecionadas

        while df_final.shape[0] < 1024:
            for j, dia in enumerate(primeiros_dias):
                for classe in [0, 1]:
                    df_dias = df_facul[(df_facul['classe'] == classe)]  
                    df_imgs = df_dias.sample(n=n_imgs[j], random_state=SEED)
                    df_final = pd.concat([df_final, df_imgs], axis=0, ignore_index=True) 

                    if df_final.shape[0] >= 1024:
                        break
                if df_final.shape[0] >= 1024:
                    break

        file_treino = f"CSV/{faculdade}/{faculdade}_Segmentado_Treino.csv"
        df_final = df_final[['caminho_imagem', 'classe']]
        df_final.to_csv(file_treino, index=False)
        print(f"DataFrame da faculdade {faculdade} salvo em {file_treino}")

        # Resetando o df_final para a próxima faculdade
        df_final = pd.DataFrame()

    #Validação
    for i, faculdade in enumerate(faculdades):
        df_facul = df[(df['Faculdade'] == f'{faculdade}')] 

        primeiros_dias = dias_cada_facul[i][:1]  
        print("O(s) respectivo(s) dia(s) foram selecionado(s) para validação: ", primeiros_dias)
        dias_cada_facul[i] = dias_cada_facul[i][1:] #Removendo os dias selecionadas

        while df_final.shape[0] < 64:
            for dia in primeiros_dias:
                for classe in [0, 1]:
                    df_dias = df_facul[(df_facul['classe'] == classe)]  
                    df_imgs = df_dias.sample(n=32, random_state=SEED)
                    df_final = pd.concat([df_final, df_imgs], axis=0, ignore_index=True) 

                    if df_final.shape[0] >= 64:
                        break
                if df_final.shape[0] >= 64:
                    break

        file_val = f"CSV/{faculdade}/{faculdade}_Segmentado_Validacao.csv"
        df_final = df_final[['caminho_imagem', 'classe']]
        df_final.to_csv(file_val, index=False)
        print(f"DataFrame da faculdade {faculdade} salvo em {file_val}")

        # Resetando o df_final para a próxima faculdade
        df_final = pd.DataFrame()

    #Teste
    for i, faculdade in enumerate(faculdades):
        df_facul = df[(df['Faculdade'] == f'{faculdade}')] 

        print("O(s) respectivo(s) dia(s) foram selecionado(s) para validação: ", dias_cada_facul[i])

        df_final  = df_facul[(df_facul['Dia'].isin(dias_cada_facul[i]))]

        file_teste = f"CSV/{faculdade}/{faculdade}_Segmentado_Teste.csv"
        df_final = df_final[['caminho_imagem', 'classe']]

        df_final.to_csv(file_teste, index=False)
        print(f"DataFrame da faculdade {faculdade} salvo em {file_teste}")

        # Resetando o df_final para a próxima faculdade
        df_final = pd.DataFrame()

PKLot()
