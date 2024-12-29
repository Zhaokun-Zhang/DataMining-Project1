from Models import Model
import os, time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def load_data():
    x1 = pd.read_csv('data/X_train.csv', header=None).values
    x2 = pd.read_csv('data/X_test.csv', header=None).values
    y1 = pd.read_csv('data/y_train.csv', header=None).values.reshape(-1)
    y2 = pd.read_csv('data/y_test.csv', header=None).values.reshape(-1)
    return x1, y1, x2, y2

def plot_scatter(data, labels, save_name):
    color_dict = {
        0:'#333333',
        1:'#4E79A7',
        2:'#F28E2B',
        3:'#E15759',
        4:'#59A14F', # green
        5:'#FFD700',
        6:'#333333', # 深灰色
    }
    plt.figure(figsize=(8,8))
    colors = pd.Series(labels).apply(lambda s:color_dict[s])
    plt.scatter(data[:,0], data[:,1], c=colors, s=1)
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{save_name}.png')

def calculate_metrics(data, labels):
    s1 = silhouette_score(data, labels)
    s2 = calinski_harabasz_score(data, labels)
    s3 = davies_bouldin_score(data, labels)
    return {
        'silhouette_score':s1,
        'calinski_harabasz_score':s2,
        'davies_bouldin_score':s3,
    }

def main():
    models = ['kmeans', 'agglomerative', 'minibatch_kmeans']
    params = {'n_clusters':6}
    trainx, trainy, testx, testy = load_data()
    os.makedirs('results', exist_ok=True)

    # actual Metrics
    actual_metrics_dict = dict()
    actual_metrics_dict['train_metrics'] = calculate_metrics(trainx, trainy)
    actual_metrics_dict['test_metrics'] = calculate_metrics(testx, testy)
    pd.DataFrame(actual_metrics_dict).to_csv('results/acutal_metrics.csv')

    

    # visualization
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, random_state=114514)
    pca_trainx = pca.fit_transform(trainx)
    tsne_trainx = tsne.fit_transform(trainx)
    pca_testx = pca.fit_transform(testx)
    tsne_testx = tsne.fit_transform(testx)

    # actual scatter
    plot_scatter(pca_trainx, trainy, save_name='PCA-train-acutal')
    plot_scatter(tsne_trainx, trainy, save_name='TSNE-train-acutal')
    plot_scatter(pca_testx, testy, save_name='PCA-test-acutal')
    plot_scatter(tsne_testx, testy, save_name='TSNE-test-acutal')


    results = dict()
    for model_name in models:
        print(model_name)
        model = Model(model_name=model_name, params=params)
        sta = time.time()
        pred_trainy = model.fit_predict(trainx)
        mid = time.time()
        pred_testy = model.predict(testx)
        end = time.time()

        result = {'train_'+k:v for k,v in calculate_metrics(trainx, pred_trainy).items()}
        result.update({'test_'+k:v for k,v in calculate_metrics(testx, pred_testy).items()})
        result.update({'train_time':mid-sta, 'inference_time':end-mid})
        results[model_name] = result
        pd.DataFrame(pred_trainy).to_csv(f'results/train_pred_y-{model_name}.csv', index=None, header=None)
        pd.DataFrame(pred_testy).to_csv(f'results/test_pred_y-{model_name}.csv', index=None, header=None)
        
        plot_scatter(pca_trainx, pred_trainy, save_name=f'PCA-train-{model_name}')
        plot_scatter(tsne_trainx, pred_trainy, save_name=f'TSNE-train-{model_name}')
        plot_scatter(pca_testx, pred_testy, save_name=f'PCA-test-{model_name}')
        plot_scatter(tsne_testx, pred_testy, save_name=f'TSNE-test-{model_name}')

    pd.DataFrame(results).T.to_csv('results/All_metrics.csv')



if __name__ == '__main__':
    main()
