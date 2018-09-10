import numpy as np

def main(datafilename):
    matrix_data = get_matrix_data_from_csv(datafilename)
    A = gen_A_matrix(matrix_data)
    P = gen_P_matrix(A)
    return

def gen_P_matrix(A):
    S, E, V = np.linalg.svd(A)
    r, c = V.shape
    p_v = V[:,(c-1)]
    return p_v.reshape(4,3)

def get_matrix_data_from_csv(datafilename):
    return np.genfromtxt(datafilename, delimiter=',', dtype=int)

def gen_A_matrix(matrix_data):
    r = matrix_data.shape[0]
    A = []
    for i in range(r):
        A.append([0,0,0,0,-(matrix_data[i][3]),-(matrix_data[i][4]),-(matrix_data[i][5]),-(matrix_data[i][6]),matrix_data[i][1]*(matrix_data[i][3]),matrix_data[i][1]*(matrix_data[i][4]),matrix_data[i][1]*(matrix_data[i][5]),matrix_data[i][1]*(matrix_data[i][6])])
        A.append([(matrix_data[i][3]),(matrix_data[i][4]),(matrix_data[i][5]),(matrix_data[i][6]),0,0,0,0,-matrix_data[i][0]*(matrix_data[i][3]),-matrix_data[i][0]*(matrix_data[i][4]),-matrix_data[i][0]*(matrix_data[i][5]),-matrix_data[i][0]*(matrix_data[i][6])])
    A = np.asarray(A, dtype=int)
    return A

if __name__ == '__main__':
    main('lego.csv')
