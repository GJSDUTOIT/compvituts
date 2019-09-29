import numpy
import statistics
import cv2 as cv
import scipy # use numpy if scipy unavailable
import scipy.linalg # use numpy if scipy unavailable
from scipy.spatial import distance
numpy.set_printoptions(threshold=numpy.inf)

def display_matches(matches):
    for match in matches:
        point1 = (int(match[0]),int(match[1]))
        point2 = (int(match[2]), int(match[3]))
        #draw point 1
        cv.circle(img, point1, 3, (255,0,0), 2)
        #draw point 2
        cv.circle(img, point2, 3, (0,0,255), 2)
        #draw line
        cv.line(img, point1, point2, (0,255,0), 2)

def threshold_outliers(matches):
    edistances = []
    for match in matches:
        point1 = (int(match[0]),int(match[1]))
        point2 = (int(match[2]), int(match[3]))
        edistances.append(distance.euclidean(point1, point2))
    sorted(edistances)
    q1, q3= numpy.percentile(edistances,[25,75])
    iqr = q3 - q1
    upper_bound = q3 +(1.5 * iqr)
    print(upper_bound)
    refined_matches = []
    for match in matches:
        point1 = (int(match[0]), int(match[1]))
        point2 = (int(match[2]), int(match[3]))
        if (distance.euclidean(point1, point2) <= upper_bound):
            refined_matches.append(match)
    refined_matches = numpy.asarray(refined_matches)
    return refined_matches

def scale_image(image):
    scale_percent = 35 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return resized

def display_thresholding_difference():
    img = cv.imread('fountain1.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    matches = numpy.loadtxt("matches.txt")
    display_matches(matches)
    cv.imwrite("/home/gjsdt/Git_Repositories/compvi-364/2019/tut4/fountain_all_matches.jpg", img)
    img = cv.imread('fountain1.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    matches = threshold_outliers(matches)
    display_matches(matches)
    cv.imwrite("/home/gjsdt/Git_Repositories/compvi-364/2019/tut4/fountain_filtered_matches.jpg", img)

def gen_ransac_fundamental_matrix(data):
    model = FundamentalMatrixModel();
    ransac_fit, ransac_data = ransac(data,model,8, 1000, 9, 5,debug=False,return_all=True)
    final_inlier_data = data[ransac_data[:],:]
    final_fundamental_matrix = model.fit(final_inlier_data)
    return final_fundamental_matrix, final_inlier_data;


def ransac(data,model,n,k,t,d,debug=False,return_all=False):
    iterations = 0
    bestfit = None
    besterr = numpy.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n,data.shape[0])
        maybeinliers = data[maybe_idxs,:]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error( test_points, maybemodel)
        # select indices of rows with accepted points
        also_idxs = []
        for i in range(len(test_err)):
            if test_err[i] < t:
                also_idxs.append(test_idxs[i])
        also_idxs = numpy.asarray(also_idxs)        
        #also_idxs = test_idxs[test_err < t]
        if also_idxs.size != 0:
            alsoinliers = data[also_idxs,:]
        if debug:
            print 'test_err.min()',test_err.min()
            print 'test_err.max()',test_err.max()
            print 'numpy.mean(test_err)',numpy.mean(test_err)
            print 'iteration %d:len(alsoinliers) = %d'%(
                iterations,len(alsoinliers))
        if len(alsoinliers) > d:
            betterdata = numpy.concatenate( (maybeinliers, alsoinliers) )
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error( betterdata, bettermodel)
            thiserr = numpy.mean( better_errs )
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = numpy.concatenate( (maybe_idxs, also_idxs) )
        iterations+=1
    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    if return_all:
        
        return bestfit, best_inlier_idxs
    else:
        return bestfit

    
def random_partition(n,n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = numpy.arange( n_data )
    numpy.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

class FundamentalMatrixModel:
    
    def fit(self, data):
        A = numpy.zeros(shape=(data.shape[0],9));
        #construct matrix A -----------------------
        for i in range(data.shape[0]):
            x_a = data[i,0];
            y_a = data[i,1];
            x_b = data[i,2];
            y_b = data[i,3];
            
            A[i,0] = x_a * x_b;
            A[i,1] = y_a * x_b;
            A[i,2] = x_b;
            A[i,3] = x_a * y_b;
            A[i,4] = y_a * y_b;
            A[i,5] = y_b;
            A[i,6] = x_a;
            A[i,7] = y_a;
            A[i,8] = 1;
        #construct matrix A ------------------------
        u_A, s_A, vh_A = numpy.linalg.svd(A, full_matrices=True);
        v_A = vh_A.T;
        f = v_A[:,-1]
        F_cap = [[f[0],f[1],f[2]],[f[3],f[4],f[5]],[f[6],f[7],f[8]]];
        u_F_cap, s_F_cap, vh_F_cap = numpy.linalg.svd(F_cap, full_matrices=True);
        #force matrix to be of rank 2
        s_F_cap[2] = 0;
        s_mat = numpy.diag(s_F_cap);
        #reconstruct F
        F = numpy.dot(u_F_cap, numpy.dot(s_mat, vh_F_cap));
        return F;
        
    def get_error(self, data, model):
        n_data_points = data.shape[0];
        error_list = numpy.zeros(shape=(n_data_points,1));
        for i in range(n_data_points):
            x_a_vector = numpy.asarray([data[i,0], data[i,1], 1]);
            x_b_vector = numpy.asarray([data[i,2], data[i,3], 1]);
            F = model;
            Fxa = numpy.matmul(F,x_a_vector);
            Ftxb = numpy.matmul(F.T,x_b_vector)
            numerator = numpy.matmul(x_b_vector.T,numpy.matmul(F,x_a_vector))**2; 
            denomenator = ((Fxa[0])**2)+((Fxa[1])**2)+((Ftxb[0])**2)+((Ftxb[1])**2);
            error_list[i] = numerator/denomenator;
        return error_list

        
if __name__=='__main__':
    
