import math
import numpy as np
import pandas as pd
import random
from scipy.sparse import coo_matrix
import scipy
import math
np.random.seed(0)
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import csr_matrix
# from sklearn.linear_model import Ridge
import random

def q1():
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from scipy.sparse.linalg import lsqr

    # Load the data
    user_song_df = pd.read_csv('user_song.csv')
    test_df = pd.read_csv('test.csv')

    # Create mappings for user_id and song_id to a sequential number for sparse matrix construction
    all_user_ids = pd.concat([user_song_df['user_id'], test_df['user_id']]).unique()
    all_song_ids = pd.concat([user_song_df['song_id'], test_df['song_id']]).unique()
    user_mapping = {id: i for i, id in enumerate(all_user_ids)}
    song_mapping = {id: i + len(user_mapping) for i, id in enumerate(all_song_ids)}

    # Append test data to user_song data with weight 0
    test_df['weight'] = np.nan
    all_data_df = pd.concat([user_song_df, test_df], ignore_index=True)

    # Calculate r_avg
    r_avg = all_data_df.loc[all_data_df['weight'].notna(), 'weight'].mean()

    # Subtract average weight from user_song ratings where weight is not 0
    all_data_df.loc[all_data_df['weight'].notna(), 'weight'] -= r_avg

    # Create lists to store row indices, column indices and data for constructing sparse matrix
    row_ind, col_ind, data = [], [], []

    for idx, row in all_data_df.iterrows():
        user_idx = user_mapping[row['user_id']]
        song_idx = song_mapping[row['song_id']]

        row_ind.extend([idx, idx])
        col_ind.extend([user_idx, song_idx])
        data.extend([1, 1])

    # Create sparse matrix
    A = sparse.csr_matrix((data, (row_ind, col_ind)))

    c = all_data_df['weight'].fillna(0).values

    b, istop, itn, r1norm, r2norm = lsqr(A, c)[:5]
    # Divide b into bu and bi
    bu = b[:len(user_mapping)]
    bi = b[len(user_mapping):]

    # Create dictionaries for bu and bi
    bu_dict = {user_id: bu_value for user_id, bu_value in zip(user_mapping.keys(), bu)}
    bi_dict = {song_id: bi_value for song_id, bi_value in zip(song_mapping.keys(), bi)}

    # Calculate biases and estimated weights for training data
    user_song_df['bu'] = user_song_df['user_id'].map(bu_dict).fillna(0)
    user_song_df['bi'] = user_song_df['song_id'].map(bi_dict).fillna(0)
    user_song_df['estimated_weight'] = r_avg + user_song_df['bu'] + user_song_df['bi']

    # If estimated weight is less than zero, set it to zero
    user_song_df['estimated_weight'] = user_song_df['estimated_weight'].clip(lower=0)

    # Calculate SSE for training data
    sse = np.sum((user_song_df.loc[user_song_df['weight'].notna(), 'weight'] - user_song_df.loc[
        user_song_df['weight'].notna(), 'estimated_weight']) ** 2)
    print("SSE for the training data:")
    print(sse)

    # Calculate biases and estimated weights for test data
    test_df['bu'] = test_df['user_id'].map(bu_dict).fillna(0)
    test_df['bi'] = test_df['song_id'].map(bi_dict).fillna(0)
    test_df['estimated_weight'] = r_avg + test_df['bu'] + test_df['bi']

    # If estimated weight is less than zero, set it to zero
    test_df['estimated_weight'] = test_df['estimated_weight'].clip(lower=0)
    test_df=test_df.drop(['Unnamed: 0','weight','bu','bi'], axis=1)
    test_df=test_df.rename(columns={'estimated_weight': 'weight' })
    test_df.to_csv('318155843_302342498_task1.csv', index=False)
    weights_train=[]
    for index, row in user_song_df.iterrows():
        user_id = row['user_id']
        song_id = row['song_id']
        actual_weight = row['estimated_weight']
        # Append the predicted rating to the weights list
        weights_train.append(actual_weight)
    return weights_train
def q2():
    import numpy as np

    def alternating_projections(user_song_matrix, max_iterations=1000, k=20, tol=300000):
        # Initialize user and item matrices P and Q
        np.random.seed(0)
        P = np.random.rand(user_song_matrix.shape[0], k)
        Q = np.random.rand(user_song_matrix.shape[1], k)

        # Variables to hold SSE values
        old_sse = np.inf
        new_sse = np.inf

        for j in range(max_iterations):
            # Update user matrix P
            for u in range(user_song_matrix.shape[0]):
                Q_R_u = Q[user_song_matrix[u, :] != 0, :]
                R_u = user_song_matrix[u, user_song_matrix[u, :] != 0]
                P[u, :] = np.linalg.lstsq(Q_R_u, R_u, rcond=None)[0]

            # Update item matrix Q
            for i in range(user_song_matrix.shape[1]):
                P_R_i = P[user_song_matrix[:, i] != 0, :]
                R_i = user_song_matrix[user_song_matrix[:, i] != 0, i]
                Q[i, :] = np.linalg.lstsq(P_R_i, R_i, rcond=None)[0]

            # Calculate SSE and check for convergence
            prediction_matrix = np.dot(P, Q.T)
            diff = user_song_matrix - prediction_matrix
            new_sse = np.sum(np.square(diff[np.where(user_song_matrix != 0)]))
            print('Iteration: {}, SSE: {}'.format(j + 1, new_sse))

            if abs(new_sse - old_sse) <= tol:
                break

            old_sse = new_sse

        return P, Q, new_sse

    user_song_df = pd.read_csv('user_song.csv')
    test_df = pd.read_csv('test.csv')
    user_song = pd.read_csv('user_song.csv')
    user_song = pd.DataFrame(user_song)
    test = pd.read_csv('test.csv')
    test = pd.DataFrame(test)
    users = list(user_song['user_id'])
    songss = list(user_song['song_id'])
    users = set(users)
    songss = set(songss)
    m = len(users)
    n = len(songss)
    train_index = []
    rate = np.zeros((m, n))
    users = np.sort(np.array(list(users)))
    songss = np.sort(np.array(list(songss)))
    for row in user_song.values:
        i = np.where(users == row[0])[0][0]
        j = np.where(songss == row[1])[0][0]
        value = row[2]
        # print(value)
        rate[i][j] = value
        train_index.append([i, j])
    P, Q, new_sse = alternating_projections(rate)
    R_PRE= np.dot(P, Q.T)
    weights = []
    for index, row in test.iterrows():
        user_id = row['user_id']
        song_id = row['song_id']
        # print("num users",len(users))
        # Find the index of the user and song in the users and songss arrays
        user_idx = np.where(users == user_id)[0][0]
        song_idx = np.where(songss == song_id)[0][0]
        # print(user_idx, song_idx)
        # Access the predicted rating from R_pred_task3
        predicted_rating = R_PRE[user_idx, song_idx]
        predicted_rating = 0 if predicted_rating < 0 else predicted_rating
        # Append the predicted rating to the weights list
        weights.append([user_idx, song_idx, predicted_rating])

    rank_test = np.zeros((m, n))
    user1 = []
    songs1 = []
    val = []
    for row in weights:
        user1.append(users[row[0]])
        songs1.append(songss[row[1]])
        val.append(row[2])
    test = {'user_id': user1, 'song_id': songs1, 'weight': val}
    test = pd.DataFrame(data=test)
    test.to_csv('318155843_302342498_task2.csv', index=False)
    print("q2 sse:",new_sse)
def q3():
    import pandas as pd
    import numpy as np
    from scipy.sparse.linalg import svds

    # Load data
    user_song_df = pd.read_csv('user_song.csv')
    user_song = pd.read_csv('user_song.csv')
    user_song = pd.DataFrame(user_song)
    test = pd.read_csv('test.csv')
    test = pd.DataFrame(test)
    users = list(user_song['user_id'])
    songss = list(user_song['song_id'])
    users = set(users)
    songss = set(songss)
    m = len(users)
    n = len(songss)
    train_index = []
    rate = np.zeros((m, n))
    users = np.sort(np.array(list(users)))
    songss = np.sort(np.array(list(songss)))
    for row in user_song.values:
        i = np.where(users == row[0])[0][0]
        j = np.where(songss == row[1])[0][0]
        value = row[2]
        # print(value)
        rate[i][j] = value
        train_index.append([i, j])
    # Pivot dataframe to get user-item matrix
    user_item_df = user_song_df.pivot(index='user_id', columns='song_id', values='weight').fillna(0)

    # Convert df to numpy array
    user_item_matrix = user_item_df.values

    # mean user ratings
    user_ratings_mean = np.mean(user_item_matrix, axis=1)
    song_ratings_mean = np.mean(user_item_matrix, axis=0)
    # De-mean (normalize by each users mean)
    user_item_demeaned = user_item_matrix - user_ratings_mean.reshape(-1, 1)

    # Compute SVD
    U, sigma, Vt = svds(user_item_demeaned, k=20)

    sigma = np.diag(sigma)

    # Predict ratings
    predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

    # Convert prediction to dataframe
    preds_df = pd.DataFrame(predicted_ratings, columns=user_item_df.columns)

    print(preds_df.head())
    # Continuing from the previous code:
    predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

    # Convert prediction to dataframe
    preds_df = pd.DataFrame(predicted_ratings, columns=user_item_df.columns, index=user_item_df.index)

    def calculate_sse(df, preds_df):
        sse = 0.0

        for index, row in df.iterrows():
            user_id = row['user_id']
            song_id = row['song_id']
            actual_weight = row['weight']
            predicted_weight = preds_df.loc[
                user_id, song_id] if user_id in preds_df.index and song_id in preds_df.columns else 0
            sse += (actual_weight - predicted_weight) ** 2
        return sse

    sse_train = calculate_sse(user_song_df, preds_df)

    print(f'The SSE on the train data in q_3 is {sse_train}')
    weights_train = []
    for index, row in user_song.iterrows():
        user_id = row['user_id']
        song_id = row['song_id']
        # Find the index of the user and song in the users and songss arrays
        user_idx = np.where(users == user_id)[0][0]
        song_idx = np.where(songss == song_id)[0][0]
        # Access the predicted rating from R_pred_task3
        predicted_rating = predicted_ratings[user_idx, song_idx]
        # Append the predicted rating to the weights list
        weights_train.append(predicted_rating)

    weights = []
    for index, row in test.iterrows():
        user_id = row['user_id']
        song_id = row['song_id']
        # print("num users",len(users))
        # Find the index of the user and song in the users and songss arrays
        user_idx = np.where(users == user_id)[0][0]
        song_idx = np.where(songss == song_id)[0][0]
        # print(user_idx, song_idx)
        # Access the predicted rating from R_pred_task3
        predicted_rating = predicted_ratings[user_idx, song_idx]
        predicted_rating = 0 if predicted_rating < 0 else predicted_rating
        # Append the predicted rating to the weights list
        weights.append([user_idx, song_idx, predicted_rating])

    rank_test = np.zeros((m, n))
    user1 = []
    songs1 = []
    val = []
    for row in weights:
        user1.append(users[row[0]])
        songs1.append(songss[row[1]])
        val.append(row[2])
    test = {'user_id': user1, 'song_id': songs1, 'weight': val}
    test = pd.DataFrame(data=test)
    test.to_csv('318155843_302342498_task3.csv', index=False)


if __name__ == '__main__':
    # Read data
    np.random.seed(0)
    user_artist = pd.read_csv('user_song.csv')
    test = pd.read_csv('test.csv')
    q1()
    # q2()
    # q3()