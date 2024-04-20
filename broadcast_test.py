import numpy as np

vec1 = list(zip(range(10),range(10)))

thresh = 1

vec1 = np.array(vec1)

print(vec1.shape)

# entity list (agent or mice)
# returns matrix of shape (len(e_list1), len(e_list2))
# where entry [i,j] tells us whether entry i and entry j 
# of their respective arrays are within given range or not 
def calc_in_range_matrix(e_list1, e_list2, range):
    # expand dims, so when we broadcast, we get difference between all pairs of values
    t1 = e_list1[:,np.newaxis,:]
    t2 = e_list2[np.newaxis,:]

    x = t1 - t2
    x = x**2
    x = x[:,:,0] + x[:,:,1]
    x = np.sqrt(x)
    x = x < range

    return x

calc_in_range_matrix(vec1,vec1,2)