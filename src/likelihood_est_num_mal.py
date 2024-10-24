import numpy as np
from scipy.optimize import minimize

def estimate_likelihood(scores_vec, score_index, num_mal):
    vec_dim = len(scores_vec[0])
    num_benign = len(scores_vec) - num_mal
    n =len(scores_vec)
    def multivariate_gaussian(param):
        mean = param[:vec_dim]
        # cov = param[vec_dim:].reshape(vec_dim, vec_dim)
        cov = param[vec_dim:]
        cov = np.diag(cov)

        x1 = scores_vec[score_index[:num_benign]]
        x2 = scores_vec[score_index[num_benign:]]
        n1 = num_benign
        n2 = num_mal

        # print(x1.shape)
        # print(mean.shape)
        # print(cov.shape)

        # ret = 0.0
        # for i in range(len(scores_vec)):
        #     x = scores_vec[i]
        #     exponent = -0.5 * np.dot(np.dot((x - mean), np.linalg.inv(cov)), (x - mean).T)
        #     coefficient = 1.0 / (np.sqrt((2 * np.pi) ** vec_dim * np.linalg.det(cov)))
        #     liklihood = np.log(coefficient) + exponent
        #     if i in score_index[:num_benign]:
        #         ret += liklihood / n
        #     else:
        #         ret -= liklihood / n

        x = scores_vec
        exponent = -0.5 * np.dot(np.dot((x - mean), np.linalg.inv(cov)), (x - mean).T)
        coefficient = 1.0 / (np.sqrt((2 * np.pi) ** vec_dim * np.linalg.det(cov)))
        liklihood = np.log(coefficient) + exponent
        liklihood = np.diagonal(liklihood)
        scalors = np.ones(len(scores_vec)) / num_benign # np.ones(len(scores_vec)) / num_benign

        scalors[score_index[num_benign:]] = -1.0 / num_mal # -1.0 / num_mal # -1
        scalors = scalors / len(scores_vec)
        ret = np.dot(liklihood, scalors)

        # exponent1 = -0.5 * np.dot(np.dot((x1 - mean), np.linalg.inv(cov)), (x1 - mean).T)
        # coefficient1 = 1.0 / (np.sqrt((2 * np.pi) ** n1 * np.linalg.det(cov)))
        # liklihood1 = coefficient1 * np.exp(exponent1)
        #
        # exponent2 = -0.5 * np.dot(np.dot((x2 - mean), np.linalg.inv(cov)), (x2 - mean).T)
        # coefficient2 = 1.0 / (np.sqrt((2 * np.pi) ** n2 * np.linalg.det(cov)))
        # liklihood2 = - coefficient2 * np.exp(exponent2)
        return -ret

    initial_mean = np.mean(scores_vec[score_index[:num_benign]],axis=0)
    # initial_cov = np.cov(scores_vec[score_index[:num_benign]]).reshape(vec_dim*vec_dim)
    initial_cov = np.ones(vec_dim,dtype=np.float)
    param = np.concatenate((initial_mean,initial_cov))
    res = minimize(multivariate_gaussian, param, method='nelder-mead', options={'xatol': 1e-8, 'disp': True, 'maxiter':200})
    # print(res.x)
    likelihood = -multivariate_gaussian(res.x)
    print(f'num_mal: {num_mal}: likelihood: {likelihood}')
    return likelihood

def estimate_likelihood_v2(scores_vec, score_index, num_mal):
    vec_dim = len(scores_vec[0])
    num_benign = len(scores_vec) - num_mal
    n =len(scores_vec)
    def multivariate_gaussian(param, cal=False):
        mean = param[:vec_dim]
        # cov = param[vec_dim:].reshape(vec_dim, vec_dim)
        cov = param[vec_dim:]
        cov = np.diag(cov)

        x1 = scores_vec[score_index[:num_benign]]
        x2 = scores_vec[score_index[num_benign:]]
        n1 = num_benign
        n2 = num_mal

        # print(x1.shape)
        # print(mean.shape)
        # print(cov.shape)

        # ret = 0.0
        # for i in range(len(scores_vec)):
        #     x = scores_vec[i]
        #     exponent = -0.5 * np.dot(np.dot((x - mean), np.linalg.inv(cov)), (x - mean).T)
        #     coefficient = 1.0 / (np.sqrt((2 * np.pi) ** vec_dim * np.linalg.det(cov)))
        #     liklihood = np.log(coefficient) + exponent
        #     if i in score_index[:num_benign]:
        #         ret += liklihood / n
        #     else:
        #         ret -= liklihood / n

        x = scores_vec
        exponent = -0.5 * np.dot(np.dot((x - mean), np.linalg.inv(cov)), (x - mean).T)
        coefficient = 1.0 / (np.sqrt((2 * np.pi) ** vec_dim * np.linalg.det(cov)))
        liklihood = np.log(coefficient) + exponent
        liklihood = np.diagonal(liklihood)
        scalors = np.ones(len(scores_vec)) / num_benign # np.ones(len(scores_vec)) / num_benign

        if cal==False:
            scalors[score_index[num_benign:]] = 0.0 # -1.0 / num_mal # -1
        else:
            scalors[score_index[num_benign:]] = -1.0 / num_mal
        scalors = scalors / len(scores_vec)
        ret = np.dot(liklihood, scalors)

        # exponent1 = -0.5 * np.dot(np.dot((x1 - mean), np.linalg.inv(cov)), (x1 - mean).T)
        # coefficient1 = 1.0 / (np.sqrt((2 * np.pi) ** n1 * np.linalg.det(cov)))
        # liklihood1 = coefficient1 * np.exp(exponent1)
        #
        # exponent2 = -0.5 * np.dot(np.dot((x2 - mean), np.linalg.inv(cov)), (x2 - mean).T)
        # coefficient2 = 1.0 / (np.sqrt((2 * np.pi) ** n2 * np.linalg.det(cov)))
        # liklihood2 = - coefficient2 * np.exp(exponent2)
        return -ret

    initial_mean = np.mean(scores_vec[score_index[:num_benign]],axis=0)
    # initial_cov = np.cov(scores_vec[score_index[:num_benign]]).reshape(vec_dim*vec_dim)
    initial_cov = np.ones(vec_dim,dtype=np.float)
    param = np.concatenate((initial_mean,initial_cov))
    res = minimize(multivariate_gaussian, param, method='nelder-mead', options={'xatol': 1e-8, 'disp': True, 'maxiter':200})
    # print(res.x)
    likelihood = -multivariate_gaussian(res.x, cal=True)
    # print(f'num_mal: {num_mal}: likelihood: {likelihood}')
    return likelihood