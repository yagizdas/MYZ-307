import csv
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

df1 = pd.read_csv("q_0_class.csv", header=None)
df2 = pd.read_csv("q_1_class.csv", header=None)
data1 = df1.values.flatten()
data2 = df2.values.flatten()

def ortalama(data):
    return sum(data)/len(data)

def standart_sapma(data):
    ort = ortalama(data)
    x = 0
    for item in data:
        x += (item - ort)**2
    x /= (len(data)-1)
    return sqrt(x)

print(np.average(data2),np.average(data1))
print(ortalama(data2),ortalama(data1))

print(standart_sapma(data2),standart_sapma(data1))
print(np.std(data2),np.std(data1))

tp,fp,fn,tn = 0,0,0,0


esik_degeri = 1

# Eşik değerinden düşük ve eşit değerler True Negative olarak toplanıyor. Aksiler ise False Negative olarak.
tn = sum(1 for item in data1 if item <= esik_degeri)
fn = sum(1 for item in data1 if item > esik_degeri)

# 1 sınıflı verilerde 1 eşik değerinin üstünde ve eşit değerler True Positive olarak değerlendirilirken aksi olanlar False Positive olarak değerlendiriliyor.
tp = sum(1 for item in data2 if item >= esik_degeri)
fp = sum(1 for item in data2 if item < esik_degeri)

print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")

confusion,confusion_list = ["True Positive", "False Positive", "False Negative", "True Negative"],[tp,fp,fn,tn]

plt.figure()
plt.bar(confusion,confusion_list)
plt.grid(True)
plt.savefig("./bar_confusion_plot.png")

precision, recall = tp/(tp+fp), tp/(tp+fn)
f1_score = tp/(tp+(fp+fn)/2)

print(f"Precision: {precision}\nRecall:{recall}\nF1 Score: {f1_score}")



