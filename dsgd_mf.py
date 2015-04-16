import sys
import math
import numpy as np
from pyspark import SparkContext, AccumulatorParam

"""
Global vars
There is no way to pass extra parameters to worker functions, so make them global
"""
n_factor = int(sys.argv[1])
n_worker = int(sys.argv[2])
n_itr = int(sys.argv[3])
beta_v = float(sys.argv[4])
lambda_v = float(sys.argv[5])

tau_v = 100     #hardcoded as 100
if_err = False   #whether calculate err each step
output_e = 'err.csv' #err path

n_user = 0
n_movie = 0
w_matrix = []
h_matrix = []

n_per_user = {}
n_per_movie = {}
user_to_row = {}
movie_to_col = {}

w_accum = 0
h_accum = 0
itr_accum = 0

total_itr_cnt = 0 #number of total sgd update
stratum = []
"""
End of global vars
"""

class Strata:
    """
    Data structure for a single strata
    """
    min_i = 0
    max_i = 0
    min_j = 0
    max_j = 0

    def __init__(self, min_i, max_i, min_j, max_j):
        self.min_i = min_i
        self.max_i = max_i
        self.min_j = min_j
        self.max_j = max_j

class FactorAccum(AccumulatorParam):
    """
    accumulator for updating w and h
    """
    def zero(self, initial):
        return np.array(0)

    def addInPlace (self, v1, v2):
        return np.add(v1, v2)

def stratum_hash(v):
    """
    Custom hash to split stratum
    """
    global user_to_row
    global n_worker
    global n_user


    i = user_to_row[v[0]]
    h = i/(n_user/n_worker)
    if h >= n_worker:
        h = n_worker - 1
    #print "hasing: " +  str(v)
    #print h

    return h

def calc_err(tuples):
    """
    Calculate avg square err
    """
    global w_matrix
    global h_matrix
    global user_to_row
    global moview_to_col

    v_matrix = np.dot(w_matrix, h_matrix)

    #print v_matrix

    err = 0
    for t in tuples:
        i = user_to_row[t[0]]
        j = movie_to_col[t[1]]
        err += math.pow(v_matrix[i][j] - t[2], 2)/len(tuples)
        #err += (v_matrix[i][j] - t[2])/len(tuples)

    return err


def line_to_tuple(text):
    """
    Convert line to tuple
    """
    for line in text:
        cols = text.strip().split(",")
        return (int(cols[0]), int(cols[1]), int(cols[2]))

def create_stratum():
    """
    Create a random stratum
    """
    global n_worker
    global n_user
    global n_movie
    global stratum

    block_h = n_user/n_worker
    block_w = n_movie/n_worker

    permutation = np.random.permutation(n_worker) #gemerate ramdom permutation of block position each row
    stratum = []  #list to store strata column in each

    #create each strata
    for i in range(n_worker):
        j = permutation[i]

        min_i = block_h*i
        min_j = block_w*j


        if i == n_worker - 1: #if last row, include all left
            max_i = n_user
        else:
            max_i = min_i + block_h

        if j == n_worker - 1: #if last col, include all left
            max_j = n_movie
        else:
            max_j = min_j + block_w

        stratum.append(Strata(min_i, max_i, min_j, max_j))

        #print min_i
        #print max_i
        #print min_j
        #print max_j
        #print " "

def check_stratum(v):
    """
    Check whether a value is in the stratum
    """
    global n_worker
    global user_to_row
    global movie_to_col
    global stratum

    #print v

    i = user_to_row[v[0]]
    j = movie_to_col[v[1]]

    for row in range(n_worker):
        strata = stratum[row]
        if (strata.min_i <= i and
            strata.max_i > i and
            strata.min_j <= j and
            strata.max_j > j):

            #print "In: " + str(v)
            return True

    #print "Out: " + str(v)
    return False

def sgd(p_tuples):
    """
    SGD update step
    """
    global beta_v
    global lambda_v
    global tau_v

    global w_matrix
    global h_matrix

    global n_per_user
    global n_per_movie
    global user_to_row
    global moview_to_col

    global w_accum
    global h_accum
    global itr_accum

    global total_itr_cnt

    delta_w_matrix = np.zeros((n_user, n_factor))
    delta_h_matrix = np.zeros((n_factor, n_movie))
    local_itr_cnt = 0 #local iteration numbers

    for t in p_tuples:
        total = total_itr_cnt + local_itr_cnt #global iteration numbers
        step_size = math.pow((tau_v + total), -beta_v)

        row = user_to_row[t[0][0]]
        col = movie_to_col[t[0][1]]

        #print "(" + str(row) + " " + str(col) + " " + str(t[1])
        #print "step size: " + str(step_size)
        #print "val :" + str(t[1])
        #print "est val: " + str(np.dot(w_matrix[row, :], h_matrix[:, col]))
        #print "w shape and h shape: " + str(w_matrix[row, :].shape) + str(h_matrix[:, col].shape)
        #print "diff: " + str(t[1] - np.dot(w_matrix[row, :], h_matrix[:, col]))


        w_step = (step_size * \
                                (-2 * (t[1] - np.dot(w_matrix[row, :], h_matrix[:, col])) * \
                                h_matrix[:, col] + \
                                2 * lambda_v / n_per_user[t[0][0]] * w_matrix[row, :]))

        h_step = (step_size * \
                                (-2 * (t[1] - np.dot(w_matrix[row, :], h_matrix[:, col])) * \
                                w_matrix[row, :] + \
                                2 * lambda_v / n_per_movie[t[0][1]] * h_matrix[:, col]))

        #print "w step: " + str(w_step);
        #print "h step: " + str(h_step);

        delta_w_matrix[row, :] -= w_step;
        delta_h_matrix[:, col] -= h_step;

        w_matrix[row, :] -= w_step;
        h_matrix[:, col] -= h_step;

        local_itr_cnt += 1

        #print t
        #print "delta_w: " + str(delta_w_matrix)
        #print "delta_h: " + str(delta_h_matrix)

    #print "step size: " + str(step_size)
    #print "local itr: " + str(local_itr_cnt)
    #print "total itr: " + str(total_itr_cnt)
    #print "delta_w: " + str(delta_w_matrix)
    #print "delta_h: " + str(delta_h_matrix)
    #print "w: " + str(w_matrix)
    #print "h: " + str(h_matrix)
    #print "next_w: " + str(w_matrix + delta_w_matrix)
    #print "next_h: " + str(h_matrix + delta_h_matrix)

    w_accum.add(delta_w_matrix)
    h_accum.add(delta_h_matrix)
    itr_accum.add(local_itr_cnt)

def main():
    global n_factor
    global n_worker
    global n_itr
    global beta_v
    global lambda_v

    global n_user
    global n_movie
    global w_matrix
    global h_matrix

    global n_per_user
    global n_per_movie
    global user_to_row
    global moview_to_col

    global w_accum
    global h_accum
    global itr_accum

    global total_itr_cnt

    input_v = sys.argv[6]
    output_w = sys.argv[7]
    output_h = sys.argv[8]

    #spark context
    sc = SparkContext("local[" + str(n_worker) + "]", "DSGD_MF")

    #read input
    text = sc.textFile(input_v)

    #convert to tuples
    tuples = text.map(line_to_tuple)

    print tuples

    #count dimension
    n_user = tuples.map(lambda t: t[0]).distinct().count()
    n_movie = tuples.map(lambda t: t[1]).distinct().count()

    #initialize w and h (randomly between 0 and 10)
    w_matrix = np.random.rand(n_user, n_factor)
    h_matrix = np.random.rand(n_factor, n_movie)
    #w_matrix = np.ones((n_user, n_factor))
    #h_matrix = np.ones((n_factor, n_movie))

    #print n_user
    #print n_movie
    #print w_matrix
    #print h_matrix

    #calculate number of movie per user and vice versa
    n_per_user = {}
    n_per_movie = {}

    tmp_user_set = tuples.map(lambda t: (t[0], 1)).reduceByKey(lambda v1, v2: v1 + v2)
    tmp_movie_set = tuples.map(lambda t: (t[1], 1)).reduceByKey(lambda v1, v2: v1 + v2)

    for u, n in tmp_user_set.collect():
        n_per_user[u] = n
    for m, n in tmp_movie_set.collect():
        n_per_movie[m] = n

    #create mapping from user/movie to actual row/col
    tmp_user_set = tuples.map(lambda t: (t[0], 1)).distinct().sortByKey().zipWithIndex()
    tmp_movie_set = tuples.map(lambda t: (t[1], 1)).distinct().sortByKey().zipWithIndex()

    for i, v in tmp_user_set.collect():
        user_to_row[i[0]] = v
    for i, v in tmp_movie_set.collect():
        movie_to_col[i[0]] = v

    #print user_to_row
    #print movie_to_col


    #setup accumulators
    w_accum = sc.accumulator(w_matrix, FactorAccum()) #for updating w
    h_accum = sc.accumulator(h_matrix, FactorAccum()) #for updating h
    itr_accum = sc.accumulator(total_itr_cnt) #for total itr numbers

    #main sgd loop
    err = [] #list for err each itr
    for itr in range(n_itr):
        create_stratum()

        #partition tuples in stratum
        stratum_tuples = tuples.filter(check_stratum).map(lambda v: ((v[0], v[1]), v[2]))
        partitioned_tuples = stratum_tuples.partitionBy(n_worker, stratum_hash).cache()

        #print stratum_tuples.count();

        #run sdg
        partitioned_tuples.foreachPartition(sgd);

        #update
        w_matrix = w_accum.value
        h_matrix = h_accum.value
        total_itr_cnt = itr_accum.value

        #compute err
        if if_err:
            err.append(calc_err(tuples.collect()))

        #print "Collected w: " + str(w_matrix)
        #print "Collected h: " + str(h_matrix)

    #save result
    np.savetxt(output_w, w_matrix, delimiter=",")
    np.savetxt(output_h, h_matrix, delimiter=",")
    #save err
    if if_err:
        np.savetxt(output_e, err, delimiter=",")
        print "Final err: " + str(err[-1])


if __name__ == '__main__':
    main()
