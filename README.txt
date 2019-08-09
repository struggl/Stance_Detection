零、任务说明
任务名称：微博话题倾向性分析
数据格式：textA textB label


一、文件说明
--data/
	数据存放路径
--log/
	模型log存放路径
--model备份_0.6334/
	最优模型备份，文件名的数字为该最优模型的指标值
--reference/
	参考文献
--Stance_Detection_v5.py
	模型类的代码，后缀v5表示这是第5个版本
--hyper_params.py
	定义模型用到的路径名和超参数
--modeling.py
	提供模型各个层的接口
	
	
二、全版本实验结果
V0:针对话题文本生成TFIDF向量,由于缺乏文本与话题之间的交互，建模并无效果

V1:仅用(条件)embed+dropout+BiLSTM+minibatch(50)+RMSprop(0.01)
	正负类平均f1:0.5212.

V2:embed+dropout+BiLSTM+Conv1d(2xW,3xW,5xW三种核各16个)+minibatch(50)+RMSprop(0.01)
	正负类平均f1:0.6079。微调以后应该可以更高一些。
	说明在条件BiLSTM基础上，加上Conv1d层，可以更好地进行话题分析建模

V3:embed+dropout+BiLSTM+Conv1d(3xW核，16个)+minibatch(50)+RMSprop(0.01)
	正负类平均f1:0.6132

V4:embed+BiLSTM+Conv1d(3xW核，16个)+dropout+SGD+RMSprop(0.001)
	正负类平均f1:0.6298
	相较于加了batch_norm的V5,峰值上暂时没有更高的表现，但是能够更稳定地保持在0.61-0.629左右.
	embed_size=32
	hidden_size=64

V5:embed+BiLSTM+Conv1d(3xW核，16个)+dropout+batch_norm+minibatch(50)+RMSprop(0.001)
	基于V4改造。
	仅添加了一句self.conv_output = modeling.normalize(self.conv_output)
	正负类平均f1:0.6334(令batch_size=1测试得到)
