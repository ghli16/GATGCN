import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import gc
import random
from clac_metric import cv_model_evaluate
from utils import *
from model import GCNModel
from opt import Optimizer

from utils import preprocess_graph, sparse_to_tuple, constructNet, constructHNet
resultList = []
resultList2 = []

def PredictScore(train_circ_dis_matrix, circ_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_circ_dis_matrix, circ_matrix, dis_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_circ_dis_matrix.sum()

    X = constructNet(train_circ_dis_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]

    features_nonzero = features[1].shape[0]

    adj_orig = train_circ_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)

    adj_nonzero = adj_norm[1].shape[0]

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_circ_dis_matrix.shape[0], name='LAGCN')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_circ_dis_matrix.shape[0], num_v=train_circ_dis_matrix.shape[1], association_nam=association_nam)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res

def circSim(circ_dis_matrix, dis_matrix):
    rows = circ_dis_matrix.shape[0]
    result = np.zeros((rows, rows))
    for i in range(0, rows):
        idx = np.where(circ_dis_matrix[i, :] == 1)
        if (np.size(idx,1)==0):
            continue
        for j in range(0, i+1):
            idy = np.where(circ_dis_matrix[j, :] == 1)
            if (np.size(idy,1)==0):
                continue
            sum1 = 0
            sum2 = 0
            for k1 in range(0, np.size(idx,1)):
                max1 = 0
                for m in range(0, np.size(idy,1)):
                    if (dis_matrix[idx[0][k1], idy[0][m]]>max1):
                        max1 = dis_matrix[idx[0][k1], idy[0][m]]
                sum1 = sum1 + max1
            for k2 in range(0, np.size(idy,1)):
                max2 = 0
                for n in range(0, np.size(idx, 1)):
                    if (dis_matrix[idx[0][n], idy[0][k2]] > max2):
                        max2 = dis_matrix[idx[0][n], idy[0][k2]]
                sum2 = sum2 + max2
            result[i, j] = (sum1 + sum2) / (np.size(idx,1) + np.size(idy,1))
            result[j, i] = result[i, j]
        for k in range(0, rows):
            result[k, k] = 1
    return result

def normalize(X):
    rows = X.shape[0]
    I = np.zeros((rows, rows))
    for i in range(0, rows):
        I[i][i] = 1
    X = X - I
    for i in range(0, rows):
        sum = X.sum(axis=1)[i]
        if (sum==0):
            continue
        for j in range(0, rows):
            X[i][j] /= sum
    return X

def cross_validation_experiment(circ_dis_matrix, dis_matrix, circ_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    index_matrix = np.mat(np.where(circ_dis_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
        random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    metric = np.zeros((1, 7))
    circ_matrix = normalize(circ_matrix)
    dis_matrix = normalize(dis_matrix)
    print("seed=%d, evaluating drug-disease...." % (seed))
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        train_matrix = np.matrix(circ_dis_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0

        circ_len = circ_dis_matrix.shape[0]
        dis_len = circ_dis_matrix.shape[1]
        circ_disease_res = PredictScore(
            train_matrix, circ_matrix*6, dis_matrix*6, seed, epochs, emb_dim, dp, lr,  adjdp)
        predict_y_proba = circ_disease_res.reshape(circ_len, dis_len)
        metric_tmp = cv_model_evaluate(
            circ_dis_matrix, predict_y_proba, train_matrix, k)
        print(metric_tmp)
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print(metric / k_folds)
    metric = np.array(metric / k_folds)
    return metric


if __name__ == "__main__":
    dis_sim = np.loadtxt('../data/dis_sim.csv', delimiter=',')
    circ_dis_matrix = np.loadtxt('../data/circRNA_disease.csv', delimiter=',')
    ci_sim = np.loadtxt('../data/circsim.csv', delimiter=',')
    epoch = 4000
    emb_dim = 64
    lr = 0.01
    adjdp = 0.3
    dp = 0.2
    simw = 6
    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    circle_time = 1
    for i in range(circle_time):
        result += cross_validation_experiment(
            circ_dis_matrix,  dis_sim, ci_sim, i, epoch, emb_dim, dp, lr, adjdp)
    average_result = result / circle_time
    np.savetxt('test/average_result.csv', average_result, delimiter=',')
    print(average_result)
