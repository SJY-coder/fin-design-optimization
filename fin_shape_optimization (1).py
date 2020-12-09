import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
# feed forward policy neural network
def build_policy_model(sizes, activation=tf.nn.relu, output_activation=tf.nn.sigmoid):
    pi_layers=[]
    for size in sizes[:-1]:
        pi_layers.append(keras.layers.Dense(size, activation=activation))
        pi_layers.append(keras.layers.BatchNormalization())
    pi_layers.append(keras.layers.Dense(sizes[-1], activation=output_activation))
    pi_model = keras.Sequential(pi_layers)
    return pi_model

def fin_shape_env(parameters=[10, 5, 5, 5, 10],done=False):
    area = 100
    lamda = 1
    h_low = 1
    h_high = 1
    a, b, c, d, e = parameters
    epsilon=1e-9
    if a<0 or b<0 or c<0 or d<0 or e<0:
        done=True
    tmp_area = (area - (a * b + (e * (c + d)) / 2))
    heat_flux = (1 / (b / (a * lamda+epsilon) + 1 / (a * h_high+epsilon) + 1 / (
                (a - c + d + 2 * np.sqrt(e ** 2 + ((d - c) / 2) ** 2)) * h_low+epsilon)))
    reward = 10*heat_flux + tmp_area
    #reward = (heat_flux+epsilon)/(tmp_area/10+epsilon)
    obs = parameters+[reward]
    return obs,reward,done,heat_flux,tmp_area


def vpg(hidden_policy_sizes=[32, 64], episode_length=3, epochs=600, render=False, lr=1e-3, ep1len = 10):
    pts=[]
    hf=[]
    ta=[]
    ms=[]
    obs_dim = 6
    n_acts = 5
    pi_model = build_policy_model(sizes=hidden_policy_sizes + [n_acts])

    gamma = 0.99
    optimizer = tf.keras.optimizers.Nadam(lr)

    def train_one_epoch(ep_loss, ep_reward,epi,ep1len=ep1len):
        with tf.GradientTape() as tape:
            for eplen in range(episode_length):
                obs_batch = []
                reward_batch = []
                action_batch = []
                log_phi_a_batch = []
                heat_flux_batch=[]
                tmp_area_batch=[]
                obs = [10, 5, 5, 5, 10,1]
                done = False
                finished_rendering = False
                for ep1len in range(ep1len):
                    if (not finished_rendering) and render:
                        print(obs[5])
                    obs = np.array(obs).reshape(-1, obs_dim)

                    obs_batch.append(obs.copy())

                    logits = pi_model(obs)

                    print(logits)

                    action = tf.random.categorical(logits, num_samples=1)

                    print(action)

                    action_ = tf.reshape(action, [-1])

                    action_mask = tf.one_hot(action_, depth=n_acts)
                    if np.sum(action_mask*logits)>=0:
                        obs[:,:5] = obs[:,:5]+np.array(action_mask*logits).reshape(1,5)
                    else:
                        obs[:,:5] = obs[:,:5]+np.array(action_mask*logits).reshape(1,5)

                    log_phi = tf.nn.log_softmax(logits)

                    log_phi_a = action_mask * log_phi

                    parameters = np.ndarray.tolist(obs.reshape(1,6))[:5]
                    obs_t1, r, done ,heat_flux,tmp_area = fin_shape_env(parameters=parameters[0][:5],done=False)
                    if done:
                        break
                    action_batch.append(action)
                    reward_batch.append(r)
                    heat_flux_batch.append(heat_flux)
                    tmp_area_batch.append(tmp_area)
                    log_phi_a_batch.append(log_phi_a)

                    obs = obs_t1
                    if ep1len==9:
                        print(obs)
                ep_len = len(reward_batch)
                eplen += 1
                g_ = []
                for i in range(ep_len - 1, -1, -1):
                    zeros_ = [0] * (i)
                    gs_ = [gamma ** j for j in reversed(range(ep_len - i))]
                    g_.append(gs_ + zeros_)

                reward = tf.constant(reward_batch, dtype=tf.float32, shape=[1, ep_len])

                gamma_ = tf.constant(g_, dtype=tf.float32)

                dis_reward_ = tf.matmul(reward, gamma_)

                log_phi_a_batch_ = tf.transpose(tf.concat(log_phi_a_batch, 0))

                loss_ = tf.reduce_sum(log_phi_a_batch_ * dis_reward_)

                total_episode_reward = tf.reduce_sum(reward_batch)
                last_reward = reward_batch[-1]
                lhf = heat_flux_batch[-1]
                lta = tmp_area_batch[-1]
                ep_loss.append(loss_)
                ep_reward.append(total_episode_reward)
                ep_reward_2.append(last_reward)
            loss = tf.math.negative(tf.reduce_mean(ep_loss))
        grads = tape.gradient(loss, pi_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, pi_model.trainable_variables))
        epoch_end_shape = obs_batch[-1]
        print(epoch_end_shape)
        if epi&60==0:
            a,b,c,d,e = epoch_end_shape[0][:5]
            ms.append([a,b,c,d,e])
            pts.append(np.array([[0,0],[0,a],[b,a],[b,a-(a-c)/2],[b+e,a-(a-d)/2],[b+e,(a-d)/2],[b,(a-c)/2],[b,0]]))
            hf.append(heat_flux_batch[-1])
            ta.append(tmp_area_batch[-1])

        print('update LOL', np.mean(ep_reward))


    ep_reward_history = []
    for epi in range(epochs):
        ep_loss = []
        ep_reward = []
        ep_reward_2 = []
        train_one_epoch(ep_loss, ep_reward,epi)
        ep_reward_history.append(np.mean(ep_reward_2))
        print(epi)

    rows = 2
    cols = 5
    axes = []
    fig = plt.figure()

    for a in range(rows * cols):
        b = Polygon(pts[a], closed=False)
        axes.append(fig.add_subplot(rows, cols, a + 1))
        subplot_title = ("Heat Flux: " + str(hf[a])[:4]+',area diff:'+str(ta[a])[:4])
        axes[-1].set_title(subplot_title)
        axes[-1].text(3, 8, 'a:'+str(ms[a][0])[:4]+'b:'+str(ms[a][1])[:4]+'c:'+str(ms[a][2])[:4]+'d:'+str(ms[a][3])[:4]+'e:'+str(ms[a][4])[:4], style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        ax = plt.gca()
        ax.add_patch(b)
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 20)
    fig.tight_layout()
    plt.show()

    plt.show()
    plt.plot(ep_reward_history)
    plt.xlabel(episode_length)
    plt.show()

vpg(hidden_policy_sizes=[16, 32, 64], episode_length=4, epochs=600, render=False, lr=2e-4, ep1len = 25)
