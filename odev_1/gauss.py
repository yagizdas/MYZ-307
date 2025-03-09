from random import gauss
import matplotlib.pyplot as plt
import csv

n_of_samples = 200

class_0,class_1 = [],[] 
avg_0_class, sd_0_class = 0, 1
avg_1_class, sd_1_class = 2, 1

for i in range (n_of_samples):
    class_0.append(gauss(avg_0_class,sd_0_class))
    class_1.append(gauss(avg_1_class,sd_1_class))

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(class_0, bins = 50,alpha=0.7, color="Blue", label="0 Sinifi")
plt.legend()

plt.subplot(1,2,2) 
plt.hist(class_1, bins = 50,alpha=0.7, color="Red", label="1 Sinifi")
plt.legend()

plt.savefig('./q_histograms.png', dpi=300)


with open("q_0_class.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(class_0)

with open("q_1_class.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(class_1)

