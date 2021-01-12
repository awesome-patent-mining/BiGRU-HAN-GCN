#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network_GCNGRU_5_head
from utils_4_withoutNodeFeature import *
import scipy.sparse as sp
import cPickle as pkl
#from tensorflow.contrib.tensorboard.plugins import projector

#SAMPLE_NUM = 17422;
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '.','path to store summary')

#change the name to who you want to send
#tf.app.flags.DEFINE_string('wechat_name', 'Tang-24-0325','the user you want to send info to')
tf.app.flags.DEFINE_string('wechat_name', 'filehelper','the user you want to send info to')

#if you want to try itchat, please set it to True
itchat_run = False
if itchat_run:
	import itchat

def main(_):
	# the path to save models
	# 获取GRU模型所需要的信息
	save_path = './model/'

	print 'reading wordembedding'
	wordembedding = np.load('./data/vec.npy')

	print 'reading training data'

	train_y = np.load('./data/train_y.npy')
	test_y = np.load('./data/testall_y.npy')
	train_sample_num = len(train_y)
	test_sample_num = len(test_y)
	# 注意这里，我要处理的是没有离群点的训练样本和测试样本
	# 所以需要把sub_graph_list.data给读进来，sub_graph_list.data中存放没有离群点的样本序号列表，这个
	# 样本序号是样本在包含离群点的样本集合中的顺序
	sub_graph_list = pkl.load(open('./patentData/sub_graph_list.data','r'))
	train_sub_graph_list = []
	test_sub_graph_list = []
	for i in sub_graph_list:
		if i<train_sample_num:
			train_sub_graph_list.append(i)
		else:
			test_sub_graph_list.append(i-train_sample_num)


	labels = np.vstack([train_y,test_y])

	labels = labels[sub_graph_list]
	#train_sen_id = np.load('./data/train_x_id_initElement.npy')
	train_word = np.load('./data/train_word.npy')
	train_pos1 = np.load('./data/train_pos1.npy')
	train_pos2 = np.load('./data/train_pos2.npy')

	train_word = train_word[train_sub_graph_list]

	train_pos1 = train_pos1[train_sub_graph_list]

	train_pos2 = train_pos2[train_sub_graph_list]

	train_y = train_y[train_sub_graph_list]

	settings = network_GCNGRU_5_head.Settings()
	settings.vocab_size = len(wordembedding)
	settings.num_classes = len(train_y[0])

	big_num = settings.big_num

	#获取GCN模型所需要的信息
	adj = pkl.load(open('./patentData/DelOutLiers_adj_withoutWeight_threshold_0.data', 'r'))
	adj_1 = pkl.load(open('./patentData/DelOutLiers_adj_withWeight_threshold_0.data', 'r'))
	SAMPLE_NUM = adj.shape[0]
	features = sp.eye(SAMPLE_NUM,dtype=np.float32).tolil()
	#sp.eye(y.shape[0] + ty.shape[0]).tolil()
	features = preprocess_features(features)
	support = preprocess_adj(adj)
	support_weight = preprocess_adj(adj_1)

	with tf.Graph().as_default():

		sess = tf.Session()
		with sess.as_default():

			initializer = tf.contrib.layers.xavier_initializer()
			with tf.variable_scope("model", reuse=None, initializer = initializer):
				m = network_GCNGRU_5_head.GRUNGCN_5_head(features,is_training=True, word_embeddings = wordembedding, settings = settings)
			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(0.001)

			#train_op=optimizer.minimize(m.total_loss,global_step=global_step)			
			train_op=optimizer.minimize(m.final_loss,global_step=global_step)			
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver(max_to_keep=None)

			merged_summary = tf.summary.merge_all()
			summary_writer = tf.summary.FileWriter(FLAGS.summary_dir+'/train_loss',sess.graph)
			# 添加4GCN
			def train_step(word_batch, pos1_batch, pos2_batch,sen_id_batch, y_batch,big_num):

				feed_dict = {}
				total_shape = []
				total_num = 0
				total_word = []
				total_pos1 = []
				total_pos2 = []
				for i in range(len(word_batch)):
					total_shape.append(total_num)
					total_num += len(word_batch[i])
					for word in word_batch[i]:
						total_word.append(word)
					for pos1 in pos1_batch[i]:
						total_pos1.append(pos1)
					for pos2 in pos2_batch[i]:
						total_pos2.append(pos2)

				total_shape.append(total_num)
				total_shape = np.array(total_shape)
				total_word = np.array(total_word)
				total_pos1 = np.array(total_pos1)
				total_pos2 = np.array(total_pos2)

				# feed_dict中的东西还挺多
				# 这是GRU的东西
				feed_dict[m.total_shape] = total_shape
				feed_dict[m.input_word] = total_word
				feed_dict[m.input_pos1] = total_pos1
				feed_dict[m.input_pos2] = total_pos2

				feed_dict[m.input_y] = y_batch
				# 这是GCN的东西
				feed_dict[m.support] = support
				feed_dict[m.support_weight] = support_weight
				feed_dict[m.input_sen_id] = sen_id_batch
				feed_dict[m.dropout] = 0.5
				feed_dict[m.num_features_nonzero] = (SAMPLE_NUM,)  # helper variable for sparse dropout
				temp, step, loss, accuracy,accuracy_1,accuracy_2,accuracy_3,accuracy_4,summary,l2_loss,final_loss= sess.run([train_op, global_step, m.total_loss, m.accuracy,m.accuracy_1,m.accuracy_2,m.accuracy_3,m.accuracy_4,merged_summary,m.l2_loss,m.final_loss], feed_dict)
				time_str = datetime.datetime.now().isoformat()
				acc = accuracy
				acc_1 = accuracy_1
				acc_2 = accuracy_2
				acc_3 = accuracy_3
				acc_4 = accuracy_4
				summary_writer.add_summary(summary,step)
				if step % 50 == 0:
					tempstr = "{}: step {}, softmax_loss {:g}, acc {:g}, acc_1 {:g},acc_2 {:g}, acc_3 {:g},acc_4 {:g}".format(time_str, step, loss, acc,acc_1,acc_2,acc_3,acc_4)
	 				#print tempstr,step,loss,acc
					print(tempstr)
	 				if itchat_run:
	 					itchat.send(tempstr,FLAGS.wechat_name)

			for one_epoch in range(settings.num_epochs):
				if itchat_run:
					itchat.send('epoch '+str(one_epoch)+' starts!',FLAGS.wechat_name)
				temp_order = range(len(train_word))
				np.random.shuffle(temp_order)
				for i in range(int(len(temp_order)/float(settings.big_num))):
					temp_word = []
					temp_pos1 = []
					temp_pos2 = []
					temp_sen_id = []
					temp_y = []
					temp_input = temp_order[i*settings.big_num:(i+1)*settings.big_num]
					for k in temp_input:
						temp_word.append(train_word[k])
						temp_pos1.append(train_pos1[k])
						temp_pos2.append(train_pos2[k])
						temp_sen_id.append(k)
						temp_y.append(train_y[k])
					num = 0
					for single_word in temp_word:
						num += len(single_word)
					
					if num > 1500:
						print 'out of range'
						continue

					temp_word = np.array(temp_word)
					temp_pos1 = np.array(temp_pos1)
					temp_pos2 = np.array(temp_pos2)
					temp_sen_id = np.array(temp_sen_id)
					temp_y = np.array(temp_y)

					train_step(temp_word,temp_pos1,temp_pos2,temp_sen_id,temp_y,settings.big_num)

					current_step = tf.train.global_step(sess, global_step)
					if current_step > 6800 and current_step%100==0:
					#if current_step == 50:
						print 'saving model'
						path = saver.save(sess,save_path +'ATT_GRU_GCN_5_head_without_Outlier_model',global_step=current_step)
						tempstr = 'have saved model to '+path
						print tempstr

			if itchat_run:
				itchat.send('training has been finished!',FLAGS.wechat_name)

if __name__ == "__main__":
	if itchat_run:
		itchat.auto_login(hotReload=True,enableCmdQR=2)
	tf.app.run() 
