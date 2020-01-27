import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_Data():
    file_data = pd.read_csv('iris.data', header = None)
    np_file = file_data.to_numpy()
    
    flwr_name = {'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    
    for row in np_file:
        row[-1] = flwr_name[row[-1]]
    # print(np_file)      
    return np_file[:, :-1], np_file[:, -1]



def initialize_centroids(dataset, num_centroids):
  centroids_list = []

  index = np.random.randint(len(dataset), size = num_centroids) # numpy array of three random numbers from 0-150

  # print(type(index))
  # print(index)

  for idx in index:
    centroids_list.append(dataset[idx])
  
  return centroids_list



def compute_Distance(p1, p2):
  return np.sqrt(np.sum(np.square(p1 - p2)))



def allocate_Points(dataset, centroids):
  distance_measures = []
    
  for point in dataset:
    # print(point)
    distance_dic = {}
    for i, centroid in enumerate(centroids):
      distance_dic[compute_Distance(point, centroid)] = i
    print(distance_dic)     
    distance_measures.append(distance_dic)

  # print(distance_dic)  
  
  alloc_list = [min(d.items())[1] for d in distance_measures]
  # print(len(alloc_list))
  return alloc_list



def shift_centroids(alloc_array, dataset):
    
  count_dict = {}
  sum_dict = {}
  data_shape = dataset[0].shape
    
  for i, point in zip(alloc_array, dataset):
    if i not in count_dict.keys():
      count_dict[i] = 1
      sum_dict[i] = point
    else:
      count_dict[i] = count_dict[i] + 1
      sum_dict[i] = sum_dict[i] + point
            
    new_centroids = [sum_dict[k]/count_dict[k] for k in sorted(sum_dict.keys())]
    new_count = [count_dict[k] for k in sorted(sum_dict.keys())]
  return new_centroids, new_count




def wcss(centroids, alloc_array, counts, dataset):
    shape = dataset[0].shape
    sum_distances = [np.zeros(shape) for _ in range(len(counts))]
    
    for ind in range(len(alloc_array)):
        sum_distances[alloc_array[ind]] = sum_distances[alloc_array[ind]] + compute_Distance(dataset[ind], centroids[alloc_array[ind]])
        
    avg_distances = [sum_distances[ind]/counts[ind] for ind in range(len(counts))]
    
    return np.sum(avg_distances) / len(avg_distances)



def plot_graph(wcss_list):
  plt.subplot(121)
  plt.plot(wcss_list, label = "WCSS")
  plt.title("WCSS Curve")
  plt.legend()
  plt.show()


# main function

np.random.seed(sum([ord(x) for x in "Iris"]))

feature_set, target_set = load_Data()
# print(feature_set) 
# print(target_set)

# initialize_centroids returns three arbitrary rows ==> three co-ordinates of points 
centroids = initialize_centroids(feature_set, 3)
# print(centroids)

old_centroids = list(centroids)
old_alloc = []

wcss_list = []

iter_count = 0

while True:
  alloc_array = allocate_Points(feature_set, centroids)
  new_centroids, counts = shift_centroids(alloc_array, feature_set)

  if old_alloc == alloc_array:
    break

  old_alloc = alloc_array
  centroids = new_centroids
  
  wcss_list.append(wcss(centroids, alloc_array, counts, feature_set))
  
  # print("BCSS score for iteration {} : {}".format(iter_count, bcss_list[-1]))
  # print("WCSS score for iteration {} : {}\n".format(iter_count, wcss_list[-1]))
  
  iter_count = iter_count + 1

# In[13]:
plot_graph(wcss_list)

# print(alloc_array)
print(centroids)


# In[23]:


# print(counts)
