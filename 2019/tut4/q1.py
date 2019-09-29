import numpy
import statistics
import cv2 as cv
import scipy # use numpy if scipy unavailable
import scipy.linalg # use numpy if scipy unavailable
from scipy.spatial import distance
numpy.set_printoptions(threshold=numpy.inf)

def display_matches():
    for match in matches:
        point1 = (int(match[0]),int(match[1]))
        point2 = (int(match[2]), int(match[3]))
        #draw point 1
        cv.circle(img, point1, 3, (255,0,0), 2)
        #draw point 2
        cv.circle(img, point2, 3, (0,0,255), 2)
        #draw line
        cv.line(img, point1, point2, (0,255,0), 2)

def threshold_outliers(threshold=0):
    global matches
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
    matches = refined_matches

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
    display_matches()
    cv.imwrite("/home/gjsdt/Git_Repositories/compvi-364/2019/tut4/fountain_all_matches.jpg", img)
    img = cv.imread('fountain1.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    threshold_outliers()
    display_matches()
    cv.imwrite("/home/gjsdt/Git_Repositories/compvi-364/2019/tut4/fountain_filtered_matches.jpg", img)

def gen_ransac_fundamental_matrix():
    matches = numpy.loadtxt("matches.txt")
    
    return


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
        also_idxs = test_idxs[test_err < t] # select indices of rows with accepted points
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
        return bestfit, {'inliers':best_inlier_idxs}
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
    
    def fit(data):
        return
        
    def get_error(data, model):
        return

class LinearLeastSquaresModel:
    """linear system solved using linear least squares

    This class serves as an example that fulfills the model interface
    needed by the ransac() function.
    
    """
    def __init__(self,input_columns,output_columns,debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    def fit(self, data):
        A = numpy.vstack([data[:,i] for i in self.input_columns]).T
        B = numpy.vstack([data[:,i] for i in self.output_columns]).T
        x,resids,rank,s = scipy.linalg.lstsq(A,B)
        return x
    def get_error( self, data, model):
        A = numpy.vstack([data[:,i] for i in self.input_columns]).T
        B = numpy.vstack([data[:,i] for i in self.output_columns]).T
        B_fit = scipy.dot(A,model)
        err_per_point = numpy.sum((B-B_fit)**2,axis=1) # sum squared error per row
        return err_per_point
        
        
def test():
    # generate perfect input data

    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20*numpy.random.random((n_samples,n_inputs) )
    perfect_fit = 60*numpy.random.normal(size=(n_inputs,n_outputs) ) # the model
    B_exact = scipy.dot(A_exact,perfect_fit)
    assert B_exact.shape == (n_samples,n_outputs)

    # add a little gaussian noise (linear least squares alone should handle this well)
    A_noisy = A_exact + numpy.random.normal(size=A_exact.shape )
    B_noisy = B_exact + numpy.random.normal(size=B_exact.shape )

    if 1:
        # add some outliers
        n_outliers = 100
        all_idxs = numpy.arange( A_noisy.shape[0] )
        numpy.random.shuffle(all_idxs)
        outlier_idxs = all_idxs[:n_outliers]
        non_outlier_idxs = all_idxs[n_outliers:]
        A_noisy[outlier_idxs] =  20*numpy.random.random((n_outliers,n_inputs) )
        B_noisy[outlier_idxs] = 50*numpy.random.normal(size=(n_outliers,n_outputs) )

    # setup model

    all_data = numpy.hstack( (A_noisy,B_noisy) )
    
    input_columns = range(n_inputs) # the first columns of the array
    output_columns = [n_inputs+i for i in range(n_outputs)] # the last columns of the array
    debug = False
    model = LinearLeastSquaresModel(input_columns,output_columns,debug=debug)

    linear_fit,resids,rank,s = scipy.linalg.lstsq(all_data[:,input_columns],
                                                  all_data[:,output_columns])

    # run RANSAC algorithm
    ransac_fit, ransac_data = ransac(all_data,model,
                                     50, 1000, 7e3, 300, # misc. parameters
                                     debug=debug,return_all=True)
    if 1:
        import pylab

        sort_idxs = numpy.argsort(A_exact[:,0])
        A_col0_sorted = A_exact[sort_idxs] # maintain as rank-2 array

        if 1:
            pylab.plot( A_noisy[:,0], B_noisy[:,0], 'k.', label='data' )
            pylab.plot( A_noisy[ransac_data['inliers'],0], B_noisy[ransac_data['inliers'],0], 'bx', label='RANSAC data' )
        else:
            pylab.plot( A_noisy[non_outlier_idxs,0], B_noisy[non_outlier_idxs,0], 'k.', label='noisy data' )
            pylab.plot( A_noisy[outlier_idxs,0], B_noisy[outlier_idxs,0], 'r.', label='outlier data' )
        pylab.plot( A_col0_sorted[:,0],
                    numpy.dot(A_col0_sorted,ransac_fit)[:,0],
                    label='RANSAC fit' )
        pylab.plot( A_col0_sorted[:,0],
                    numpy.dot(A_col0_sorted,perfect_fit)[:,0],
                    label='exact system' )
        pylab.plot( A_col0_sorted[:,0],
                    numpy.dot(A_col0_sorted,linear_fit)[:,0],
                    label='linear fit' )
        pylab.legend()
        pylab.show()

        
def gen_2Ddata():
    # generate perfect input data

    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20*numpy.random.random((n_samples,n_inputs) )
    perfect_fit = 60*numpy.random.normal(size=(n_inputs,n_outputs) ) # the model
    B_exact = scipy.dot(A_exact,perfect_fit)
    assert B_exact.shape == (n_samples,n_outputs)

    # add a little gaussian noise (linear least squares alone should handle this well)
    A_noisy = A_exact + numpy.random.normal(size=A_exact.shape )
    B_noisy = B_exact + numpy.random.normal(size=B_exact.shape )

    if 1:
        # add some outliers
        n_outliers = 100
        all_idxs = numpy.arange( A_noisy.shape[0] )
        numpy.random.shuffle(all_idxs)
        outlier_idxs = all_idxs[:n_outliers]
        non_outlier_idxs = all_idxs[n_outliers:]
        A_noisy[outlier_idxs] =  20*numpy.random.random((n_outliers,n_inputs) )
        B_noisy[outlier_idxs] = 50*numpy.random.normal(size=(n_outliers,n_outputs) )

    # setup model

    all_data = numpy.hstack( (A_noisy,B_noisy) )    
    return all_data;

    
    
if __name__=='__main__':
    data = numpy.loadtxt("matches.txt")
    n = 8;
    print(data.shape[0]);
    maybe_idxs, test_idxs = random_partition(n,data.shape[0])
    print(maybe_idxs);
    print(test_idxs);
    print("MAYBE INLIERS\n\n\n")
    maybeinliers = data[maybe_idxs,:]
    print(maybeinliers);
    print("test points\n\n\n")
    test_points = data[test_idxs]
    print(test_points);
    
    '''
    n = 8;
    data = gen_2Ddata();
    print(data.shape[0]);
    maybe_idxs, test_idxs = random_partition(n,data.shape[0])
    print(maybe_idxs);
    print(test_idxs);
    print("MAYBE INLIERS\n\n\n")
    maybeinliers = data[maybe_idxs,:]
    print(maybeinliers);
    print("test points\n\n\n")
    test_points = data[test_idxs]
    print(test_points);'''