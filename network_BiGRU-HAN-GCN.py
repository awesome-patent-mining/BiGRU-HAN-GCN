#-*-coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from layers import *
from metrics import *

class Settings(object):

	def __init__(self):

		self.vocab_size = 114042
		self.num_steps = 70
		self.num_epochs = 50
		self.num_classes = 15
		self.gru_size = 230
		self.keep_prob = 0.5
		self.num_layers = 1
		self.pos_size = 5
		self.pos_num = 123
		# the number of entity pairs of each batch during training or testing
		self.big_num = 50

#GCN_NO_LINEWEIGHT,GCN_LINEWEIGHT,BiGRU,BiGRU-GCN_LINEWEIGHT,BiGRU-GCN_NO_LINEWEIGHT
class GRUNGCN_5_head:
	def __init__(self,features, is_training, word_embeddings, settings):

		self.num_steps = num_steps = settings.num_steps
		self.vocab_size = vocab_size = settings.vocab_size
		self.num_classes = num_classes = settings.num_classes
		self.gru_size = gru_size = settings.gru_size
		self.big_num = big_num = settings.big_num

		self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_word')
		self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_pos1')
		self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_pos2')
		self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
		self.input_sen_id = tf.placeholder(dtype=tf.int32, shape=[None], name='input_sen_id')
		self.total_shape = tf.placeholder(dtype=tf.int32, shape=[big_num + 1], name='total_shape')
		total_num = self.total_shape[-1]



		word_embedding = tf.get_variable(initializer=word_embeddings, name='word_embedding')
		pos1_embedding = tf.get_variable('pos1_embedding', [settings.pos_num, settings.pos_size])
		pos2_embedding = tf.get_variable('pos2_embedding', [settings.pos_num, settings.pos_size])

		attention_w = tf.get_variable('attention_omega', [gru_size, 1])
		sen_a = tf.get_variable('attention_A', [gru_size])
		sen_r = tf.get_variable('query_r', [gru_size, 1])
		relation_embedding = tf.get_variable('relation_embedding', [self.num_classes,gru_size])
		sen_d = tf.get_variable('bias_d', [self.num_classes])

		fc_1 = tf.get_variable('fc_1', [self.num_classes+self.num_classes,self.num_classes])
		fc_bias_1 = tf.get_variable('fc_bias_1', [self.num_classes])

		fc_2 = tf.get_variable('fc_2', [self.num_classes+self.num_classes,self.num_classes])
		fc_bias_2 = tf.get_variable('fc_bias_2', [self.num_classes])

		# gru_cell_forward = tf.nn.rnn_cell.GRUCell(gru_size)
		# gru_cell_backward = tf.nn.rnn_cell.GRUCell(gru_size)
		# tf.contrib.rnn.RNNCell
		gru_cell_forward = tf.contrib.rnn.GRUCell(gru_size)
		gru_cell_backward = tf.contrib.rnn.GRUCell(gru_size)

		if is_training and settings.keep_prob < 1:
			gru_cell_forward = tf.contrib.rnn.DropoutWrapper(gru_cell_forward, output_keep_prob=settings.keep_prob)
			gru_cell_backward = tf.contrib.rnn.DropoutWrapper(gru_cell_backward, output_keep_prob=settings.keep_prob)
		# tf.contrib.rnn.DropoutWrapper
		cell_forward = tf.contrib.rnn.MultiRNNCell([gru_cell_forward] * settings.num_layers)
		cell_backward = tf.contrib.rnn.MultiRNNCell([gru_cell_backward] * settings.num_layers)

		sen_repre = []
		sen_alpha = []
		sen_s = []
		sen_out = []
		self.prob = []
		self.prob_1 = []
		self.prob_2 = []
		self.prob_3 = []
		self.prob_4 = []

		self.predictions = []
		self.predictions_1 = []
		self.predictions_2 = []
		self.predictions_3 = []
		self.predictions_4 = []
		self.loss = []
		self.accuracy = []
		self.accuracy_1 = []
		self.accuracy_2 = []
		self.accuracy_3 = []
		self.accuracy_4 = []
		self.total_loss = 0.0

		self._initial_state_forward = cell_forward.zero_state(total_num, tf.float32)
		self._initial_state_backward = cell_backward.zero_state(total_num, tf.float32)

		# embedding layer
		inputs_forward = tf.concat([tf.nn.embedding_lookup(word_embedding, self.input_word),
									tf.nn.embedding_lookup(pos1_embedding, self.input_pos1),
									tf.nn.embedding_lookup(pos2_embedding, self.input_pos2)], 2)
		inputs_backward = tf.concat([tf.nn.embedding_lookup(word_embedding, tf.reverse(self.input_word, [1])),
									 tf.nn.embedding_lookup(pos1_embedding, tf.reverse(self.input_pos1, [1])),
									 tf.nn.embedding_lookup(pos1_embedding, tf.reverse(self.input_pos2, [1]))], 2)

		outputs_forward = []

		state_forward = self._initial_state_forward

		# Bi-GRU layer
		with tf.variable_scope('GRU_FORWARD'):
			for step in range(num_steps):
				if step > 0:
					tf.get_variable_scope().reuse_variables()
				(cell_output_forward, state_forward) = cell_forward(inputs_forward[:, step, :], state_forward)
				outputs_forward.append(cell_output_forward)

		outputs_backward = []

		state_backward = self._initial_state_backward
		with tf.variable_scope('GRU_BACKWARD'):
			for step in range(num_steps):
				if step > 0:
					tf.get_variable_scope().reuse_variables()
				(cell_output_backward, state_backward) = cell_backward(inputs_backward[:, step, :], state_backward)
				outputs_backward.append(cell_output_backward)

		output_forward = tf.reshape(tf.concat(outputs_forward, 1), [total_num, num_steps, gru_size])
		output_backward = tf.reverse(tf.reshape(tf.concat(outputs_backward, 1), [total_num, num_steps, gru_size]), [1])

		# word-level attention layer
		output_h = tf.add(output_forward, output_backward)
		attention_r = tf.reshape(tf.matmul(tf.reshape(tf.nn.softmax(
			tf.reshape(tf.matmul(tf.reshape(tf.tanh(output_h), [total_num * num_steps, gru_size]), attention_w),
					   [total_num, num_steps])), [total_num, 1, num_steps]), output_h), [total_num, gru_size])

		# sentence-level attention layer
		for i in range(big_num):

			sen_repre.append(tf.tanh(attention_r[self.total_shape[i]:self.total_shape[i + 1]]))
			batch_size = self.total_shape[i + 1] - self.total_shape[i]

			sen_alpha.append(
				tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.multiply(sen_repre[i], sen_a), sen_r), [batch_size])),
						   [1, batch_size]))
			#GRU和GCN合并，GRU在sen_s已经足够了

			sen_s.append(tf.reshape(tf.matmul(sen_alpha[i],sen_repre[i]),[gru_size,1]))
			sen_out.append(tf.add(tf.reshape(tf.matmul(relation_embedding,sen_s[i]),[self.num_classes]),sen_d))

		sen_out = tf.convert_to_tensor(sen_out)
		sen_out = tf.reshape(sen_out, (self.big_num,self.num_classes))
		#添加GCN_noWeight相关部分开始
		self.layers = []
		self.activations = []
		# support就是处理后的邻接矩阵
		self.support = tf.sparse_placeholder(tf.float32)
		# 由于GCN的特点，训练集和测试集的feature连在一起加入运算
		#self.features = tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),

		self.inputs = tf.SparseTensor(indices= features[0], values= features[1], dense_shape = features[2])
		self.input_dim = features[2][1]
		self.GCNhidden1 = 16

		self.vars = {}

		self.dropout = tf.placeholder_with_default(0., shape=()),
		#注意，上面这个dropout和keep probability是一回事
		self.num_features_nonzero = tf.placeholder(tf.int32)  # helper variable for sparse dropout
		self.placeholders = {
			'support':self.support,
			'dropout':self.dropout,
			'num_features_nonzero':self.num_features_nonzero
		}

		with tf.variable_scope('GCN_noWeight'):
			self._build()

		# Build sequential layer model
		self.activations.append(self.inputs)
		for layer in self.layers:
			hidden = layer(self.activations[-1])
			self.activations.append(hidden)
		self.outputs = self.activations[-1]
		#我需要从self.outputs中mask一下，抽取batch中的内容

		gcn_output = tf.gather(params=self.outputs, indices=self.input_sen_id)
		#--GCN_noWeight添加结束


		# 添加GCN_Weight相关部分开始
		self.layers_1 = []
		self.activations_1 = []
		# support就是处理后的邻接矩阵
		self.support_weight = tf.sparse_placeholder(tf.float32)
		# 由于GCN的特点，训练集和测试集的feature连在一起加入运算

		self.inputs_1 = tf.SparseTensor(indices=features[0], values=features[1], dense_shape=features[2])
		self.input_dim_1 = features[2][1]
		self.GCNhidden1_1 = 16

		self.vars_1 = {}

		self.dropout_1 = tf.placeholder_with_default(0., shape=()),
		# 注意，上面这个dropout和keep probability是一回事
		self.num_features_nonzero_1 = tf.placeholder(tf.int32)  # helper variable for sparse dropout
		self.placeholders['support_weight'] = self.support_weight

		with tf.variable_scope('GCN_weight'):
			self._build_1()

		# Build sequential layer model
		self.activations_1.append(self.inputs_1)
		for layer in self.layers_1:
			hidden = layer(self.activations_1[-1])
			self.activations_1.append(hidden)
		self.outputs_1 = self.activations_1[-1]
		# 我需要从self.outputs中mask一下，抽取batch中的内容

		gcn_output_1 = tf.gather(params=self.outputs_1, indices=self.input_sen_id)
		# --GCN_weight添加结束

		#添加第一个头，从GRU_GCN_noWeight中预测关系类型

		#然后将GCN_noWeight_batch和GRU_batch进行concat
		concat_output = tf.concat([sen_out,gcn_output],1)
		#fc_out = sen_out
		fc_out = tf.add(tf.reshape(tf.matmul(concat_output,fc_1), [self.big_num,self.num_classes]), fc_bias_1)
		#fc_out = gcn_output
		self.prob = tf.nn.softmax(fc_out)

		with tf.name_scope("output"):
			self.predictions = tf.argmax(fc_out, 1, name="predictions")

		with tf.name_scope("accuracy"):
			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.argmax(self.input_y, 1)), "float"),
							   name="accuracy")



		# 添加第二个头，用来从GRU的输出结果中预测关系类别
		self.prob_1 = tf.nn.softmax(sen_out)

		with tf.name_scope("output_1"):
			self.predictions_1 = tf.argmax(sen_out, 1, name="predictions_1")
		with tf.name_scope("accuracy_1"):
			self.accuracy_1 = tf.reduce_mean(tf.cast(tf.equal(self.predictions_1, tf.argmax(self.input_y, 1)), "float"),
							   name="accuracy_1")


		# 添加第三个头，用来从GCN的输出结果中预测关系类别
		self.prob_2 = tf.nn.softmax(gcn_output)

		with tf.name_scope("output_2"):
			self.predictions_2 = tf.argmax(gcn_output, 1, name="predictions_2")
		with tf.name_scope("accuracy_2"):
			self.accuracy_2 = tf.reduce_mean(tf.cast(tf.equal(self.predictions_2, tf.argmax(self.input_y, 1)), "float"),
							   name="accuracy_2")
		# tf.summary.scalar('loss',self.total_loss)

		# 添加第四个头，从GRU_GCN_weight中预测关系类型

		# 将GCN_weight_batch和GRU_batch进行concat
		concat_output_1 = tf.concat([sen_out, gcn_output_1], 1)

		fc_out_1 = tf.add(tf.reshape(tf.matmul(concat_output_1, fc_2), [self.big_num, self.num_classes]), fc_bias_2)
		self.prob_3 = tf.nn.softmax(fc_out_1)

		with tf.name_scope("output_3"):
			self.predictions_3 = tf.argmax(fc_out_1, 1, name="predictions_3")

		with tf.name_scope("accuracy_3"):
			self.accuracy_3 = tf.reduce_mean(tf.cast(tf.equal(self.predictions_3, tf.argmax(self.input_y, 1)), "float"),
											   name="accuracy")

		# 添加第五个头，用来从GCN_weight的输出结果中预测关系类别
		self.prob_4 = tf.nn.softmax(gcn_output_1)

		with tf.name_scope("output_4"):
			self.predictions_4 = tf.argmax(gcn_output_1, 1, name="predictions_4")
		with tf.name_scope("accuracy_4"):
			self.accuracy_4 = tf.reduce_mean(tf.cast(tf.equal(self.predictions_4, tf.argmax(self.input_y, 1)), "float"),
							   name="accuracy_2")
		# 然后计算loss
		with tf.name_scope("loss"):
			self.total_loss = tf.reduce_sum(
				tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=fc_out))
			self.total_loss = self.total_loss + tf.reduce_sum(
				tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=sen_out))
			self.total_loss = self.total_loss + tf.reduce_sum(
				tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=gcn_output))
			self.total_loss = self.total_loss +tf.reduce_sum(
				tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=fc_out_1))
			self.total_loss = self.total_loss + tf.reduce_sum(
				tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=gcn_output_1))

		tf.summary.scalar('loss', self.total_loss)
		# regularization

		self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
															  weights_list=tf.trainable_variables())
		self.final_loss = self.total_loss + self.l2_loss
		tf.summary.scalar('l2_loss', self.l2_loss)
		tf.summary.scalar('final_loss', self.final_loss)


	def build(self):
		""" Wrapper for _build() """
		with tf.variable_scope(self.name):
			self._build()

		# Build sequential layer model
		self.activations.append(self.inputs)
		for layer in self.layers:
			hidden = layer(self.activations[-1])
			self.activations.append(hidden)
		self.outputs = self.activations[-1]

		# Store model variables for easy access
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
		self.vars = {var.name: var for var in variables}

		# Build metrics
		self._loss()
		self._accuracy()
		self._output_labels()
		# self._precision()
		# self._recall()

		self.opt_op = self.optimizer.minimize(self.loss)


	def _loss(self):
		# Weight decay loss
		for var in self.layers[0].vars.values():
			self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

		# Cross entropy error
		self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
												  self.placeholders['labels_mask'])


	def _accuracy(self):
		self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
										self.placeholders['labels_mask'])


	def _precision(self):
		self.precision = masked_precision(self.outputs, self.placeholders['labels'],
										  self.placeholders['labels_mask'])


	def _recall(self):
		self.recall = masked_recall(self.outputs, self.placeholders['labels'],
									self.placeholders['labels_mask'])


	def _f1(self):
		self.f1 = masked_f1(self.outputs, self.placeholders['labels'],
							self.placeholders['labels_mask'])


	def _output_labels(self):
		self.true_labels, self.predict_labels = output_labels(self.outputs, self.placeholders['labels'],
															  self.placeholders['labels_mask'])


	def _build(self):
		self.layers.append(GraphConvolution(input_dim=self.input_dim,
											output_dim=self.GCNhidden1,
											placeholders=self.placeholders,
											act=tf.nn.relu,
											dropout=True,
											sparse_inputs=True))

		self.layers.append(GraphConvolution(input_dim=self.GCNhidden1,
											output_dim=self.num_classes,
											placeholders=self.placeholders,
											act=lambda x: x,
											dropout=True))


	def _build_1(self):
		self.layers_1.append(GraphConvolution_1(input_dim=self.input_dim,
											output_dim=self.GCNhidden1,
											placeholders=self.placeholders,
											act=tf.nn.relu,
											dropout=True,
											sparse_inputs=True))

		self.layers_1.append(GraphConvolution_1(input_dim=self.GCNhidden1,
											output_dim=self.num_classes,
											placeholders=self.placeholders,
											act=lambda x: x,
											dropout=True))
	def predict(self):
		return tf.nn.softmax(self.outputs)