import numpy as np
import pickle, os, sys, logging, time, argparse, pdb
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def load_from_pickle(pklfile):
    with open(pklfile, 'rb') as f:
        data = pickle.load(f)

    return data


def write_to_pickle(data, pklfile):
    with open(pklfile, 'wb') as f:
        pickle.dump(data, f)


def rfr_quantile_pred_ints(est, X, percentile=95):
    logger.info('Random Forest Quantile Regression')
    err_down = []
    err_up = []
    for i in range(len(X)):
        if i % 100 == 0:
            logger.info(f"{time.strftime('%y-%m-%d_%H:%M:%S')}, estimator {i}/{len(X)}")
        preds = []
        for pred in est.estimators_:
            preds.append(pred.predict(X[i].reshape(1,-1))) # had [0] but dont' think it's needed
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    logger.info('iterations finished')

    return err_down, err_up


''' random forest quantile regression script '''

if __name__=='__main__':
    parser = argparse.ArgumentParser('tool to run quantile regression on a fitted random forest model')
    parser.add_argument('--data',dest='data',help='ttest split pkl file')
    parser.add_argument('--est',dest='est',help='fitted estimator pkl file')
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_from_pickle(args.data)
    est = load_from_pickle(args.est)

    logger.info(f'data file: {os.path.basename(args.data)}')
    logger.info(f'fitted estimator: {os.path.basename(args.est)}')

    pred_ints = rfr_quantile_pred_ints(est, X_test, percentile=95)
    outfile = os.path.join(os.path.dirname(args.data),'pred_ints.pkl')
    write_to_pickle(pred_ints, outfile)
    logger.info(f'written to file: {outfile}')
