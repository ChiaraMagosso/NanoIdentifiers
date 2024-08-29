## 
## Copyright (C) 2024 by
## Chiara Magosso
## 
## This work is licensed under a  
## 
## Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
## ( http://creativecommons.org/licenses/by-nc-sa/4.0/ )
## 
## Please contact chiara.magosso@polito.it for information.
##
## This code is associated with the following work : https://doi.org/10.21203/rs.3.rs-4170364/v1
##

import skimage.color
import skimage.io
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.util import crop
import matplotlib.colors
from numpy import savetxt
import os

path = 'QR-Code_data/' 

indirizzo_FP= f'{path}Fingerprint-Enhancement/'

indirizzo_ADAb = f'{path}ADAblock/'

os.makedirs(os.path.join(path, 'analysis'), exist_ok=True)

for root, dirs, files in os.walk(indirizzo_FP):
    for file in files:
        if file.endswith(".tif") and file.startswith('B66'):
            
            estensione_imm = '.tif'
            estensione_dat = '.xls'
            immagine_sem = (f'{indirizzo_FP}{file}')
            cartella = file.replace(estensione_imm,'')
            cartella_puf = f'{path}analysis/'
            im = skimage.io.imread(immagine_sem)
            df = pd.read_csv(f'{indirizzo_ADAb}{cartella}/defect_coordinates{estensione_dat}', sep = '\t')

            dimensioni_original=im.shape
            y=dimensioni_original[0]
            x=dimensioni_original[1]
            cut_y = int((dimensioni_original[0]/10)/2)

            df = df.loc[(df['Y'] > cut_y) & (df['Y'] <= (y - cut_y)) ]  
            df['Y']=df['Y']-cut_y

            im_crop = crop(im, ((cut_y, cut_y), (0, 0), (0,0)), copy=False)
            dimensioni = im_crop.shape
            y=dimensioni[0]
            x=dimensioni[1]

            cut_x = int((x-y)/2)

            df = df.loc[(df['X'] > cut_x) & (df['X'] <= (x - cut_x)) ] 
            df['X']=df['X']-cut_x 

            im_crop_final = crop(im_crop, ((0, 0), (cut_x, cut_x), (0,0)), copy=False)

            dimensioni=im_crop_final.shape
            y=dimensioni[0]
            x=dimensioni[1]

            if x!=y:
                if x>y:
                    im_crop_final = crop(im_crop_final, ((0, 0), (0, 1), (0,0)), copy=False) 
                    df = df.loc[ (df['X'] <= (x - 1)) ]
                    dimensioni=im_crop_final.shape
                    y=dimensioni[0]
                    x=dimensioni[1]
                else:
                    im_crop_final = crop(im_crop_final, ((0, 1), (0, 0), (0,0)), copy=False) 
                    df = df.loc[ (df['Y'] <= (y-1)) ] 
                    dimensioni=im_crop_final.shape
                    y=dimensioni[0]
                    x=dimensioni[1]
            dimensioni=im_crop_final.shape
            y=dimensioni[0]
            x=dimensioni[1]
            if x!=y:
                print("CONTROLLA LE DIMENSIONI DI IMMAGINE, NON E' ANCORA UN QUADRATO")
            
            df["PhaseConnectivity"] = df['Phase'].astype(str) + df["Connectivity"].astype(str)
            markers = {0: "p", 1: "o", 3:"^", 4:"P", 5:"*", 2:"d", 6:"s"}
            palette={"00":"#FF00FF", "01":"#FFFC00", "03":"#FFD162", "04":"#FFB405", "05":"#FFA082", "10":"#00B3B3", "11":"#00EBFF", "13":"#0038A0", "14":"#1E7800", "15":"#5E16FF"}
            
            plt.figure(figsize=(6, 6), dpi = 600)
            grafico2=sns.scatterplot(x=df['X'], y=df['Y'], hue=df['PhaseConnectivity'], style=df['Connectivity'], markers=markers, palette=palette, legend=False)
            plt.imshow(im_crop_final)
            grafico2.set(xticklabels=[])  
            grafico2.set(xlabel=None)
            grafico2.tick_params(bottom=False)
            grafico2.set(yticklabels=[])  
            grafico2.set(ylabel=None)
            grafico2.tick_params(left=False)
            plt.savefig(f'{cartella_puf}{cartella}_difetti_cut.png')
            plt.clf()

            df = df.loc[ (df['Phase'] ==  0) ]
            
            grafico3=sns.scatterplot(x=df['X'], y=df['Y'], hue=df['PhaseConnectivity'], style=df['Connectivity'], markers=markers, palette=palette, legend=False)
            plt.imshow(im_crop_final)
            grafico3.set(xticklabels=[])  
            grafico3.set(xlabel=None)
            grafico3.tick_params(bottom=False)
            grafico3.set(yticklabels=[])  
            grafico3.set(ylabel=None)
            grafico3.tick_params(left=False)
            plt.savefig(f'{cartella_puf}{cartella}_difetti_cut_una_fase.png')
            plt.clf()

            df.to_excel(f'{cartella_puf}{cartella}_difetti_cut_una_fase.xlsx') 
            print(f'image : {cartella} , dimensions : ', im_crop_final.shape )
            
            bins = 10
            uno = 3
            zero = 10
                
            istogramma, xedges, yedges = np.histogram2d(df['X'], df['Y'], bins=bins, range= [[0, x], [0, y]])
            istogramma = istogramma.T

            data=istogramma.flatten()
            savetxt(f'{cartella_puf}{cartella}_isto_data.csv', istogramma, delimiter=';')

            plt.imshow(istogramma, origin='upper', cmap='Greys_r')
            plt.colorbar()
            plt.savefig(f'{cartella_puf}{cartella}_isto.png')
            plt.clf()
            istogramma_normalizzato = np.where(istogramma!=0, 1, 0)
            uno=np.count_nonzero(istogramma_normalizzato)
            zero=istogramma_normalizzato.size-np.count_nonzero(istogramma_normalizzato)

            data=istogramma_normalizzato.flatten()
            savetxt(f'{cartella_puf}{cartella}_isto_norm_data.csv', istogramma_normalizzato, delimiter=';')

            cmap = matplotlib.colors.ListedColormap(['black','w'])
            plt.imshow(istogramma_normalizzato, origin='upper', cmap=cmap)
            plt.colorbar()
            plt.savefig(f'{cartella_puf}{cartella}_isto_norm.png')
            plt.clf()

            tipi_difetti = df['Connectivity'].unique()
            colori = ['r','g','b','c','m','y']
            my_list = []
            for tipo, colore in zip(tipi_difetti,colori):
                
                difetto = df.loc[(df['Connectivity'] == tipo ) ] 

                istogramma, xedges, yedges = np.histogram2d(difetto['X'], difetto['Y'], bins=bins, range= [[0, x], [0, y]])
                istogramma = istogramma.T

                data=istogramma.flatten()
                savetxt(f'{cartella_puf}{cartella}_isto_data_difetto={tipo}.csv', istogramma, delimiter=';')

                plt.imshow(istogramma, origin='upper', cmap='cool')
                plt.colorbar()
                plt.savefig(f'{cartella_puf}{cartella}_isto_difetto={tipo}.png')
                plt.clf()
                istogramma_normalizzato = np.where(istogramma!=0, 1, 0)

                uno=np.count_nonzero(istogramma_normalizzato)
                zero=istogramma_normalizzato.size-np.count_nonzero(istogramma_normalizzato)

                data=istogramma_normalizzato.flatten()
                savetxt(f'{cartella_puf}{cartella}_isto_norm_data_difetto={tipo}.csv', istogramma_normalizzato, delimiter=';')
                
                my_list.extend(data)

                cmap = matplotlib.colors.ListedColormap(['black', colore])
                plt.imshow(istogramma_normalizzato, origin='upper', cmap=cmap)
                plt.colorbar()
                plt.savefig(f'{cartella_puf}{cartella}_isto_norm_difetto={tipo}.png')
                plt.clf()
            
            liste = np.array_split(my_list, len(my_list)/100)
            count=1
            pezzo=0
            while pezzo< len(liste):
                liste[pezzo] = liste[pezzo] * pow(10, count)
                count += 1
                pezzo += 1

            res = np.sum(liste, 0).reshape(10, 10)
            plt.imshow(res, origin='upper', cmap='viridis')
            plt.colorbar(spacing = 'proportional').set_ticks([0, 10, 100, 110, 1000, 1010, 1100, 1110])
            plt.title(f'all different defect type isto normalizzato')
            plt.savefig(f'{cartella_puf}{cartella}_isto_norm_difetti_diversi.png')
            plt.clf()
            data=res.flatten()
            savetxt(f'{cartella_puf}{cartella}_isto_norm_data_difetti_diversi.csv', res, delimiter=';')