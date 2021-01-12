# -*- coding: utf-8 -*-
import numpy as np
import os

#embedding the position 
def pos_embed(x):
	if x < -60:
		return 0
	if x >= -60 and x <= 60:
		return x+61
	if x > 60:
		return 122
#find the index of x in y, if x not in y, return -1
def find_index(x,y):
	flag = -1
	for i in range(len(y)):
		if x != y[i]:
			continue
		else:
			return i
	return flag
#注意，在处理过程中，原始数据有丢失，查看下什么原因
#reading data
def init():
	
	print 'reading word embedding data...'
	vec = []
	word2id = {}
	f = open('./origin_data/vec.txt')
	f.readline()
	while True:
		content = f.readline()
		if content == '':
			break
		content = content.strip().split()
		word2id[content[0]] = len(word2id)
		content = content[1:]
		content = [(float)(i) for i in content]
		vec.append(content)
	f.close()
	word2id['UNK'] = len(word2id)
	word2id['BLANK'] = len(word2id)
	
	dim = 50
	vec.append(np.random.normal(size=dim,loc=0,scale=0.05))
	vec.append(np.random.normal(size=dim,loc=0,scale=0.05))
	vec = np.array(vec,dtype=np.float32)


	print 'reading relation to id'
	relation2id = {}	
	f = open('./origin_data/relation2id.txt','r')
	while True:
		content = f.readline()
		if content == '':
			break
		content = content.strip().split()
		relation2id[content[0]] = int(content[1])
	f.close()

	#length of sentence is 70
	fixlen = 70
	#max length of position embedding is 60 (-60~+60)
	maxlen = 60

	train_sen = {} #{entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
	train_ans = {} #{entity pair:[label1,label2,...]} the label is one-hot vector

	print('reading test data ...')

	test_sen = {} #{entity pair:[[sentence 1],[sentence 2]...]}
	test_ans = {} #{entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)
	test_sen_id = {}
	f = open('./origin_data/trainset_AR_RE_13Thousand_4gcn.txt','r')
	#设置个计数器，记录每个记录的序号，方便在testall_word.npy和quanlet中的数据进行对应
	idx_counter = -1;
	while True:
		idx_counter = idx_counter + 1;
		content = f.readline()
		if content == '':
			break
		
		content = content.strip().split()
		en1 = content[0]
		en2 = content[1]
		relation = 0
		if content[2] not in relation2id:
			raise RuntimeError(content[2]+' is not in relation dictionary!')
		else:
			relation = relation2id[content[2]]
		tup = (en1,en2)
		
		if tup not in test_sen:
			test_sen[tup]=[]
			test_sen_id[tup] = []
			y_id = relation
			label_tag = 0
			label = [0 for i in range(len(relation2id))]
			label[y_id] = 1
			test_ans[tup] = label
		else:
			y_id = relation
			test_ans[tup][y_id] = 1
			
		sentence = content[3:-1]

		en1pos = 0
		en2pos = 0
		
		for i in range(len(sentence)):
			if sentence[i] == en1:
				en1pos = i
			if sentence[i] == en2:
				en2pos = i
		output = []

		for i in range(fixlen):
			word = word2id['BLANK']
			rel_e1 = pos_embed(i - en1pos)
			rel_e2 = pos_embed(i - en2pos)
			output.append([word,rel_e1,rel_e2])

		for i in range(min(fixlen,len(sentence))):
			word = 0
			if sentence[i] not in word2id:
				word = word2id['UNK']
			else:
				word = word2id[sentence[i]]

			output[i][0] = word
		test_sen[tup].append(output)
		test_sen_id[tup].append(idx_counter)

	test_x = []
	test_y = []
	test_x_id = []


	print 'organizing test data'
	f = open('./data/test_q&a.txt','w')
	temp=0
	for i in test_sen:		
		test_x.append(test_sen[i])
		test_x_id.append(test_sen_id[i])
		test_y.append(test_ans[i])
		tempstr = ''
		for j in range(len(test_ans[i])):
			if test_ans[i][j]!=0:
				tempstr = tempstr+str(j)+'\t'
		#print str(temp)+'\t'+i[0]+'\t'+i[1]+'\t'+tempstr+'\n'
		f.write(str(temp)+'\t'+i[0]+'\t'+i[1]+'\t'+tempstr+'\n')
		#2	thin/film/type/magnetic/head	conductor	0	2
		temp+=1
	f.close()

	test_x = np.array(test_x)
	test_y = np.array(test_y)
	test_x_id = np.array(test_x_id)

	#np.save('./data/vec.npy',vec)

	np.save('./data/testall_x.npy',test_x)
	np.save('./data/testall_y.npy',test_y)
	np.save('./data/testall_x_id.npy', test_x_id)


def seperate():
	print 'seperating test all data'
	x_test = np.load('./data/testall_x.npy')

	test_word = []
	test_pos1 = []
	test_pos2 = []

	for i in range(len(x_test)):
		word = []
		pos1 = []
		pos2 = []
		for j in x_test[i]:
			temp_word = []
			temp_pos1 = []
			temp_pos2 = []
			for k in j:
				temp_word.append(k[0])
				temp_pos1.append(k[1])
				temp_pos2.append(k[2])
			word.append(temp_word)
			pos1.append(temp_pos1)
			pos2.append(temp_pos2)
		test_word.append(word)
		test_pos1.append(pos1)
		test_pos2.append(pos2)



	test_word = np.array(test_word)
	test_pos1 = np.array(test_pos1)
	test_pos2 = np.array(test_pos2)

	np.save('./data/testall_word.npy',test_word)
	np.save('./data/testall_pos1.npy',test_pos1)
	np.save('./data/testall_pos2.npy',test_pos2)


#get answer metric for PR curve evaluation
def getans():
	test_y = np.load('./data/testall_y.npy')
	eval_y = []
	for i in test_y:
		eval_y.append(i[1:])
	allans = np.reshape(eval_y,(-1))
	np.save('./data/allans.npy',allans)

def get_metadata():
	fwrite = open('./data/metadata.tsv','w')
	f = open('./origin_data/vec.txt')
	f.readline()
	while True:
		content = f.readline().strip()
		if content == '':
			break
		name = content.split()[0]
		fwrite.write(name+'\n')
	f.close()
	fwrite.close()


init()
seperate()
getans()
get_metadata()

