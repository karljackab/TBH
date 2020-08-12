import tensorflow as tf
from tensorflow.python.framework import function
import util.layers as layers
from util.dataset import DataHelper, MatDataset, BasicDataset, InferenceDataHelper
from time import gmtime, strftime
import os
from meta import REPO_PATH
import matplotlib.pyplot as plt


@function.Defun(tf.float32, tf.float32, tf.float32, tf.float32)
def doubly_sn_grad(logits, epsilon, dprev, dpout):
    prob = 1.0 / (1 + tf.exp(-logits))
    dlogits = prob * (1 - prob) * (dprev + dpout)
    depsilon = dprev
    return dlogits, depsilon

@function.Defun(tf.float32, tf.float32, grad_func=doubly_sn_grad)
def doubly_sn(logits, epsilon):
    prob = 1.0 / (1 + tf.exp(-logits))
    yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
    return yout, prob

class TBH(object):
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size', 100)
        self.code_length = kwargs.get('code_length', 32)
        self.continue_length = kwargs.get('continue_length', 512)
        self.input_length = kwargs.get('input_length', 2048)
        self.middle_dim = kwargs.get('middle_dim', 1024)
        self.LR = kwargs.get('LR', 1e-4)
        self.iteration = kwargs.get('iteration', 100000)
        self.output_folder_name = kwargs.get('output_folder_name', strftime("%a%d%b%Y-%H%M%S", gmtime()))
        self.epsilon = 1e-7

        self.image_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_length])
        self.lam = kwargs.get('lam', 1)
        self.net = self._build_net()
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    @staticmethod
    def fc_layer_hack(name, bottom, input_dim, output_dim, bias_term=True, weights_initializer=None,
                  biases_initializer=None):
        # flatten bottom input
        # input has shape [batch, in_height, in_width, in_channels]
        flat_bottom = bottom

        # weights and biases variables
        with tf.compat.v1.variable_scope(name):
            # initialize the variables
            if weights_initializer is None:
                weights_initializer = tf.random_normal_initializer(stddev=0.01)
            if bias_term and biases_initializer is None:
                biases_initializer = tf.constant_initializer(0.)

            # weights has shape [input_dim, output_dim]
            weights = tf.compat.v1.get_variable("kernel", [input_dim, output_dim], initializer=weights_initializer)
            if bias_term:
                biases = tf.compat.v1.get_variable("bias", output_dim, initializer=biases_initializer)
                fc = tf.compat.v1.nn.xw_plus_b(flat_bottom, weights, biases)
            else:
                fc = tf.matmul(flat_bottom, weights)
        return fc

    def _build_net(self):
        with tf.compat.v1.variable_scope('actor'):
            with tf.compat.v1.variable_scope('encoder'):
                ## first stage of encoder
                fc_1 = layers.fc_relu_layer('fc_1', bottom=self.image_in, output_dim=self.middle_dim)
                ## generate continuous latent variable
                continuous_hidden = layers.fc_relu_layer('cont_hidden', bottom=fc_1, output_dim=self.continue_length)
                
                ## generate binary latent variable
                code_hidden = layers.fc_layer('code_hidden', bottom=fc_1, output_dim=self.code_length)
                _batch_size, _feature_size = self.image_in.get_shape().as_list()
                eps = tf.ones([_batch_size, self.code_length], dtype=tf.float32) * 0.5
                ### change code_hidden to binary hashing vector
                codes, _ = doubly_sn(code_hidden, eps)
                codes.set_shape([_batch_size, self.code_length])

                ## biulding adjacency matrix
                batch_adjacency = layers.build_adjacency_hamming(codes, code_length=self.code_length)

                ## compute GCN, output latent vector z'
                hidden_z = tf.nn.sigmoid(
                    layers.spectrum_conv_layer('gcn', continuous_hidden, batch_adjacency, self.continue_length, _batch_size))

            with tf.compat.v1.variable_scope('decoder'):
                ## decode latent vector
                fc_2 = layers.fc_relu_layer('fc_2', hidden_z, self.middle_dim)
                decode_result = layers.fc_relu_layer('decode_result', fc_2, _feature_size)

        ## discriminator - real part
        with tf.compat.v1.variable_scope('critic'):
            real_logic = tf.sigmoid(layers.fc_layer('critic', bottom=hidden_z, output_dim=1),
                                    name='critic_sigmoid')

            real_binary_logic = tf.sigmoid(
                self.fc_layer_hack('critic_2', bottom=codes, input_dim=self.code_length, output_dim=1),
                name='critic_sigmoid_2')

        ## discriminator - fake part
        with tf.compat.v1.variable_scope('critic', reuse=True):
            random_in = tf.random.uniform([_batch_size, self.continue_length])
            fake_logic = tf.sigmoid(layers.fc_layer('critic', bottom=random_in, output_dim=1),
                                    name='critic_sigmoid')

            random_binary = (tf.sign(tf.random.uniform([_batch_size, self.code_length]) - 0.5) + 1) / 2
            fake_binary_logic = tf.sigmoid(
                self.fc_layer_hack('critic_2', bottom=random_binary, input_dim=self.code_length, output_dim=1),
                name='critic_sigmoid_2')

        return {
            'codes': codes,
            'code_hidden': code_hidden,
            'decode_result': decode_result,
            'cont_hidden': continuous_hidden,
            'hidden': hidden_z,
            'real_logic': real_logic,
            'fake_logic': fake_logic,
            'real_binary_logic': real_binary_logic,
            'fake_binary_logic': fake_binary_logic}

    def _build_loss(self):
        ## discriminator loss
        critic_loss_1 = tf.reduce_mean(tf.math.log(self.net.get('fake_logic') + self.epsilon)
                            + tf.math.log(1 - self.net.get('real_logic') + self.epsilon)
                            ) * -1 * self.lam

        critic_loss_2 = tf.reduce_mean(
            tf.math.log(self.net.get('fake_binary_logic') + self.epsilon) 
            + tf.math.log(1 - self.net.get('real_binary_logic') + self.epsilon)
            ) * -1 * self.lam
        
        critic_loss = critic_loss_1 + critic_loss_2

        ## encoder loss
        encoding_loss = tf.reduce_mean(tf.nn.l2_loss(
            self.net.get('decode_result') - self.image_in)) - self.lam * tf.reduce_mean(
            tf.math.log(self.net.get('real_logic') + self.epsilon)) - self.lam * tf.reduce_mean(
            tf.math.log(self.net.get('real_binary_logic') + self.epsilon))

        return encoding_loss, critic_loss

    def opt(self, operator: tf.Tensor, scope=None):
        ## use Adam and beta1=0.9, beta2=0.999 to minimize upcoming target
        train_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return tf.compat.v1.train.AdamOptimizer(self.LR, beta1=0.9, beta2=0.999).minimize(operator, global_step=self.global_step, var_list=train_list)

    @staticmethod
    def plot_loss(x, y, title, save_pth):
        ## plot curve of input x and y to save_pth
        plt.xlabel('time')
        plt.ylabel('loss')
        plt.title(title)
        plt.plot(x, y)
        plt.savefig(save_pth)
        plt.close()

    def train(self, sess: tf.compat.v1.Session, data: DataHelper, restore_file=None, log_path='data'):
        ## prepare loss term (actor loss and critic loss)
        actor_loss, critic_loss = self._build_loss()
        actor_opt = self.opt(actor_loss, 'actor')
        critic_opt = self.opt(critic_loss, 'critic')

        ## initialization
        initial_op = tf.compat.v1.global_variables_initializer()
        sess.run(initial_op)

        ## prepare storing path (model weight path and log path)
        save_path = os.path.join(REPO_PATH, log_path, 'model', self.output_folder_name)
        summary_path = os.path.join(REPO_PATH, log_path, 'log', self.output_folder_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)

        ## check whether to restore weight
        if restore_file is not None:
            self._restore(sess, restore_file)

        actor_loss_list, critic_loss_list, tot_loss_list = [], [], []   ## loss record list
        x_axis = [] ## x axis of loss curve
        with open(os.path.join(summary_path, 'actor_loss'), 'w') as f_actor, \
                open(os.path.join(summary_path, 'critic_loss'), 'w') as f_critic, \
                open(os.path.join(summary_path, 'tot_loss'), 'w') as f_tot:
            ## start training and optimize loss function
            for i in range(self.iteration):
                ## get current batch input
                train_batch = data.next_batch()
                train_dict = {self.image_in: train_batch['batch_image']}

                ## feed input to model, and optimize the loss function (critic_opt, actor_opt)
                ## get loss value for visualization (critic_value, actor_value)
                _, critic_value, _ = sess.run(
                    [critic_opt, critic_loss, self.global_step], feed_dict=train_dict)
                _, actor_value, code_value, _ = sess.run(
                    [actor_opt, actor_loss, self.net['codes'], self.global_step], feed_dict=train_dict)

                ## record the result code of this batch, for following MAP calculation
                data.update(code_value)

                if (i + 1) % 100 == 0:
                    hook_train = data.hook_train()  ## get training MAP
                    print('batch {}: actor {}, critic {}, MAP {}'.format(i, actor_value, critic_value, hook_train))
                    
                    ## write loss and MAP to file
                    f_actor.write(f'{i}:{actor_value}\n')
                    f_critic.write(f'{i}:{critic_value}\n')
                    f_tot.write(f'{i}:{actor_value+critic_value}\n')

                    ## append to list, for following visualization
                    actor_loss_list.append(actor_value)
                    critic_loss_list.append(critic_value)
                    tot_loss_list.append(actor_value+critic_value)
                    x_axis.append(i)

                ## save the model every 3000 iter
                if (i + 1) % 3000 == 0:
                    self._save(sess, save_path, i+1)
        
        ## visualize loss curve
        self.plot_loss(x_axis, actor_loss_list, 'actor_loss', os.path.join(summary_path, 'actor_loss.png'))        
        self.plot_loss(x_axis, critic_loss_list, 'critic_loss', os.path.join(summary_path, 'critic_loss.png'))     
        self.plot_loss(x_axis, tot_loss_list, 'tot_loss', os.path.join(summary_path, 'tot_loss.png'))     

    @staticmethod
    def _restore(sess: tf.compat.v1.Session, restore_file, var_list=None):
        ## restore specific model checkpoint
        print(f'restore file {restore_file}')
        saver = tf.compat.v1.train.Saver(var_list=var_list)
        saver.restore(sess, save_path=restore_file)
        print('restore successfully')

    @staticmethod
    def _save(sess: tf.compat.v1.Session, save_path, step):
        ## store specific model checkpoint
        saver = tf.compat.v1.train.Saver()
        save_path = os.path.join(save_path, 'model')
        saver.save(sess, save_path, step)
        print('Saved!')

    def extract(self, sess: tf.compat.v1.Session, data: InferenceDataHelper, 
                restore_file=None, folder='data/code', task='cifar', 
                phase='test', suffix=None):
        ## generate the code the data, and store it to file
        if restore_file is not None:
            self._restore(sess, restore_file)

        for _ in range(data.data.batch_num + 1):
            this_batch = data.next_batch()
            this_dict = {self.image_in: this_batch['batch_image']}
            code = sess.run(self.net['codes'], feed_dict=this_dict)
            data.update(code)

        data.save(task, self.code_length, phase=phase, folder=folder, suffix=suffix)
