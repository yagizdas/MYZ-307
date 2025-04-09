import cv2
import numpy as np
import os 

def data_loader(file_path1, file_path2):
    filenames_1, filenames_2= os.listdir("odev_3/"+ file_path1),  os.listdir("odev_3/" + file_path2)
    folders, images_list, folder_name = (filenames_1,filenames_2), [], (file_path1, file_path2)
    print(folders)

    for i in range(len(folders)):
        for file in folders[i]:
            im = np.array(cv2.imread("odev_3/"+folder_name[i]+file, cv2.IMREAD_GRAYSCALE))
            print(im.shape)
            images_list.append(im.flatten())
    return np.array(images_list)

collected_data = data_loader("face1/", "face2/")

print(collected_data.shape)

xeksimean = collected_data - np.mean(collected_data)
covariance = np.dot(xeksimean,xeksimean.T)/((collected_data.shape[0])-1)

print(np.cov(collected_data).shape)
print("el yapımı")
print(covariance.shape)


eigenvalues, eigenvectors = np.linalg.eigh(covariance)

print(eigenvalues)
print(eigenvectors)

print(eigenvalues[-1])
print(eigenvectors[-1][:10])

h,w = 192,168

