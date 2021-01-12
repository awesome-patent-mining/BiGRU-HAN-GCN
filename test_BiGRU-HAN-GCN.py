#-*-coding:utf-8
import tensorflow as tf
import numpy as np
import time
import datetime
import os
from utils_4_withoutNodeFeature import *
import network_GCNGRU_5_head
from sklearn.metrics import average_precision_score
import cPickle as pkl
FLAGS = tf.app.flags.FLAGS
#change the name to who you want to send
#tf.app.flags.DEFINE_string('wechat_name', 'Tang-24-0325','the user you want to send info to')
tf.app.flags.DEFINE_string('wechat_name', 'filehelper','the user you want to send info to')
#if you want to try itchat, please set it to True
itchat_run = False
if itchat_run:
	import itchat

def main(_):
	# ATTENTION: change pathname before you load your model
	pathname = "./model/ATT_GRU_GCN_5_head_without_Outlier_model-"
	
	wordembedding = np.load('./data/vec.npy')

	test_settings = network_GCNGRU_5_head.Settings()
	test_settings.vocab_size = 114044
	test_settings.num_classes = 15
	test_settings.big_num = 87
	big_num_test = test_settings.big_num
	adj = pkl.load(open('patentData/DelOutLiers_adj_withoutWeight_threshold_0.data', 'r'))
	adj_1 = pkl.load(open('patentData/DelOutLiers_adj_withWeight_threshold_0.data', 'r'))
	SAMPLE_NUM = adj.shape[0]
	features = sp.eye(SAMPLE_NUM, dtype=np.float32).tolil()
	features = preprocess_features(features)
	support = preprocess_adj(adj)
	support_weight = preprocess_adj(adj_1)

	with tf.Graph().as_default():

		sess = tf.Session()
		with sess.as_default():

			def test_step(word_batch, pos1_batch, pos2_batch,sen_id_batch, y_batch):
	
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

				feed_dict[mtest.total_shape] = total_shape
				feed_dict[mtest.input_word] = total_word
				feed_dict[mtest.input_pos1] = total_pos1
				feed_dict[mtest.input_pos2] = total_pos2
				feed_dict[mtest.input_y] = y_batch

				# 这是GCN的东西
				feed_dict[mtest.support] = support
				feed_dict[mtest.support_weight] = support_weight
				feed_dict[mtest.input_sen_id] = sen_id_batch
				feed_dict[mtest.dropout] = 0.5
				feed_dict[mtest.num_features_nonzero] = (SAMPLE_NUM,)  # helper variable for sparse dropout

				loss, accuracy,accuracy_1 ,accuracy_2,accuracy_3 ,accuracy_4,prob,prob_1,prob_2,prob_3,prob_4= sess.run(
					[mtest.loss, mtest.accuracy,mtest.accuracy_1,mtest.accuracy_2,mtest.accuracy_3,mtest.accuracy_4,mtest.prob,mtest.prob_1,mtest.prob_2,mtest.prob_3,mtest.prob_4], feed_dict)
				return prob,prob_1,prob_2,prob_3,prob_4,accuracy,accuracy_1,accuracy_2,accuracy_3,accuracy_4

			
			with tf.variable_scope("model"):
				mtest = network_GCNGRU_5_head.GRUNGCN_5_head(features,is_training=False, word_embeddings = wordembedding, settings = test_settings)
			saver = tf.train.Saver()
			# ATTENTION: change the list to the iters you want to test !!
			testlist = [6900]
			# 所以需要把sub_graph_list.data给读进来，sub_graph_list.data中存放没有离群点的样本序号列表，这个
			# 样本序号是样本在包含离群点的样本集合中的顺序
			sub_graph_list = pkl.load(open('./patentData/sub_graph_list.data', 'r'))
			train_sub_graph_list = []
			test_sub_graph_list = []
			train_y = np.load('./data/train_y.npy')
			test_y = np.load('./data/testall_y.npy')
			train_sample_num = len(train_y)
			test_sample_num = len(test_y)
			for i in sub_graph_list:
				if i < train_sample_num:
					train_sub_graph_list.append(i)
				else:
					test_sub_graph_list.append(i - train_sample_num)
			for model_iter in testlist:
				saver.restore(sess,pathname+str(model_iter))
				print("Evaluating for iter "+str(model_iter))

				test_word = np.load('./data/testall_word.npy')
				test_pos1 = np.load('./data/testall_pos1.npy')
				test_pos2 = np.load('./data/testall_pos2.npy')

				test_word = test_word[test_sub_graph_list]

				test_pos1 = test_pos1[test_sub_graph_list]

				test_pos2 = test_pos2[test_sub_graph_list]

				test_y = test_y[test_sub_graph_list]

				#train_sen_id = np.load('./data/train_x_id_initElement.npy')
				#test_sen_id = np.load('./data/testall_x_id_initElement.npy')
				test_sen_id = range(len(train_sub_graph_list),len(train_sub_graph_list)+len(test_sub_graph_list))

				#ChenLiang add some codes,all relationships includes NoEdge
				allprob_GRUGCN = []
				allprob_GRUGCN_1 = []
				allprob_GRUGCN_2 = []
				allprob_GRUGCN_3 = []
				allprob_GRUGCN_4 = []
				#acc = []
				for i in range(int(len(test_word)/float(test_settings.big_num))):
					prob,prob_1,prob_2,prob_3,prob_4,accuracy,accuracy_1,accuracy_2,accuracy_3,accuracy_4 = test_step(test_word[i*test_settings.big_num:(i+1)*test_settings.big_num],test_pos1[i*test_settings.big_num:(i+1)*test_settings.big_num],test_pos2[i*test_settings.big_num:(i+1)*test_settings.big_num],test_sen_id[i*test_settings.big_num:(i+1)*test_settings.big_num],test_y[i*test_settings.big_num:(i+1)*test_settings.big_num])
					#acc.append(np.mean(np.reshape(np.array(accuracy),(test_settings.big_num))))
					prob = np.reshape(np.array(prob),(test_settings.big_num,test_settings.num_classes))
					prob_1 = np.reshape(np.array(prob_1), (test_settings.big_num, test_settings.num_classes))
					prob_2 = np.reshape(np.array(prob_2), (test_settings.big_num, test_settings.num_classes))
					for single_prob in prob:
						allprob_GRUGCN.append(single_prob)
					for single_prob in prob_1:
						allprob_GRUGCN_1.append(single_prob)
					for single_prob in prob_2:
						allprob_GRUGCN_2.append(single_prob)
					for single_prob in prob_3:
						allprob_GRUGCN_3.append(single_prob)
					for single_prob in prob_4:
						allprob_GRUGCN_4.append(single_prob)
				print 'saving all test result...'
				current_step = model_iter
				# ATTENTION: change the save path before you save your result !!
				#np.save('./out/testset_2thousand_allprob_plusNoEdge_iter_' + str(current_step) + '.npy', allprob_plusNoEdge)
				np.save('./data/without_outlier_testall_y.npy',test_y)
				np.save('./out/testset_GRU_GCN_5_head_without_Outlier_iter_' + str(current_step) + '.npy',allprob_GRUGCN)
				np.save('./out/testset_GRU_GCN_5_head_without_Outlier_iter_' + str(current_step) + '_1.npy', allprob_GRUGCN_1)
				np.save('./out/testset_GRU_GCN_5_head_without_Outlier_iter_' + str(current_step) + '_2.npy',
						allprob_GRUGCN_2)
				np.save('./out/testset_GRU_GCN_5_head_without_Outlier_iter_' + str(current_step) + '_3.npy',
						allprob_GRUGCN_3)
				np.save('./out/testset_GRU_GCN_5_head_without_Outlier_iter_' + str(current_step) + '_4.npy',
						allprob_GRUGCN_4)
				time_str = datetime.datetime.now().isoformat()
				print time_str

if __name__ == "__main__":
	if itchat_run:
		itchat.auto_login(hotReload=True,enableCmdQR=2)
	tf.app.run() 
