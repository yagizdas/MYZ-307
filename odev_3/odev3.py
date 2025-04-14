from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
TOTAL_IMG_FOLDER = "C:/Users/PC/Desktop/myz/odev_3/faces/"

train_set_files = os.listdir(TOTAL_IMG_FOLDER)

width  = 168
height = 192
print('Görseller:')
train_image_names = os.listdir(TOTAL_IMG_FOLDER)
training_tensor   = np.ndarray(shape=(len(train_image_names), height*width), dtype=np.float64)

print(training_tensor.shape)
fig, axes = plt.subplots(5,2, figsize=(5,13))
for i in range(len(train_image_names)):
    img = plt.imread(TOTAL_IMG_FOLDER + train_image_names[i])
    training_tensor[i,:] = np.array(img, dtype='float64').flatten()
    row = i // 2
    col = i % 2
    axes[row,col].set_title(i)
    axes[row,col].imshow(img, cmap='gray')
    axes[row,col].tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()


mean_face = np.zeros((1,height*width))

for i in training_tensor:
    mean_face = np.add(mean_face,i)

mean_face = np.divide(mean_face,float(len(train_image_names))).flatten()

plt.imshow(mean_face.reshape(height, width), cmap='gray')
plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.title("Ortalama Yüz")
plt.show()
#X - Xmean kısmı

normalised_training_tensor = np.ndarray(shape=(len(train_image_names), height*width))

for i in range(len(train_image_names)):
    normalised_training_tensor[i] = np.subtract(training_tensor[i],mean_face)
fig,axes = plt.subplots(5,2, figsize=(5,13))
for i in range(len(train_image_names)):
    img = normalised_training_tensor[i].reshape(height,width)
    row = i // 2
    col = i % 2

    axes[row,col].imshow(img, cmap='gray')
    axes[row,col].tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()
cov_matrix=np.cov(training_tensor)

print('Covariance Matrix Shape:', cov_matrix.shape)

print(cov_matrix)
eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
print('Özdeğer boyutları: {} Özvektör boyutları: {}\n'.format(eigenvalues.shape, eigenvectors.shape))

print("Özdeğerler:\n", eigenvalues,"\n")
print("Özvektörler:\n",eigenvectors)
eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

eig_pairs.sort(reverse=True)
eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

sorted_ind = sorted(range(eigenvalues.shape[0]), key=lambda k: eigenvalues[k], reverse=True)

eigvalues_sort = eigenvalues[sorted_ind]
eigvectors_sort = eigenvectors[sorted_ind]

var_comp_sum = np.cumsum(eigvalues_sort)/sum(eigvalues_sort)

print("Cumulative proportion of variance explained vector: \n%s" %var_comp_sum)

num_comp = range(1,len(eigvalues_sort)+1)
plt.title('Cum. Prop. Variance Explain and Components Kept')
plt.xlabel('Principal Components')
plt.ylabel('Cum. Prop. Variance Expalined')

plt.scatter(num_comp, var_comp_sum)
plt.show()


reduced_data = np.array(eigvectors_sort[:5]).transpose()
print(reduced_data.shape)
print(training_tensor.transpose().shape, reduced_data.shape)
proj_data = np.dot(training_tensor.transpose(),reduced_data)
proj_data = proj_data.transpose()
proj_data.shape
for i in range(proj_data.shape[0]):
    img = proj_data[i].reshape(height,width)
    plt.subplot(5,5,1+i)
    plt.title(f"{i+1}")
    plt.imshow(img, cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')

plt.show()

k = 10 

eigenfaces = np.dot(normalised_training_tensor.T, eigvectors_sort[:k].T)  

for i in range(eigenfaces.shape[1]):
    eigenfaces[:, i] /= np.linalg.norm(eigenfaces[:, i])

W = eigenfaces

X_centered = training_tensor - mean_face
train_projections = np.dot(X_centered, W)  


print(X_centered.shape)
print(train_projections.shape)
# Yeniden yapılandırılmış görüntüleri saklamak için liste
reconstructed_images = []  

#öklid uzaklığı oluşturmak için kullanılan liste
reconstruction_errors = []

num_images = train_projections.shape[0]

# Her bir görüntü için yeniden yapılandırma işlemi:
for i in range(num_images):

    # Proje katsayıları (shape: (k,))
    proj_coeff = train_projections[i]

    recon_vector = mean_face + np.dot(W, proj_coeff)
    recon_image = recon_vector.reshape(height, width)
    reconstructed_images.append(recon_image)

    error = np.linalg.norm(training_tensor[i] - recon_vector)
    reconstruction_errors.append(error)

# Görselleştirme (örneğin 2 satır 5 sütunluk grid şeklinde)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()
for i, ax in enumerate(axes):
    if i < num_images:
        ax.imshow(reconstructed_images[i], cmap='gray')
        ax.set_title(f"Image {i}")
    ax.axis('off')
plt.suptitle("Yeniden Yapılandırılmış Görüntüler")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
print(reconstruction_errors)
