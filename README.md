# DeepLog
A Pytorch implementation of [DeepLog](https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf)'s log key anomaly detection model.

*If you are confusing about how to extract log key (i.e. log template), I recommend using **Drain** which is proposed in this [paper](https://pinjiahe.github.io/papers/ICWS17.pdf). As far as I know, it is the most effective log parsing method. By the way, there is a toolkit and benchmarks for automated log parsing in this [repository](https://github.com/logpai/logparser).*

## Requirement
* python>=3.6
* pytorch==1.4
* tensorboard==2.0.2

## Dataset
~~The dataset can be downloaded [HERE](https://www.cs.utah.edu/~mind/papers/deeplog_misc.html).~~ The website can't accessed now, but you can find the HDFS data in this repository.

The original HDFS logs can be found [HERE] (http://people.iiis.tsinghua.edu.cn/~weixu/sospdata.html).

## Visualization
Run the following code in terminal, then navigate to https://localhost:6006.

`tensorboard --logdir=log`

## Reference
Min Du, Feifei Li, Guineng Zheng, Vivek Srikumar. "Deeplog: Anomaly detection and diagnosis from system logs through deep learning." ACM SIGSAC Conference on Computer and Communications Security(CCS), 2017.

## Contributing
**If you have any questions, please open an** ***[issue](https://github.com/wuyifan18/DeepLog/issues).***

**Welcome to** ***[pull requests](https://github.com/wuyifan18/DeepLog/pulls)*** **to implement the rest of this paper!**



## my readme

### demo


丟近來的log label需要重1開始
把log 丟到data下面
進LogKeyModel_train.py 
    修改 num_class
    修改  seq_dataset = generate("data/openstack_normal1.log_preprocess_session_numberID") 
進LogKeyModel_predict.py 修改 num_class
    修改  seq_dataset = generate("data/openstack_normal1.log_preprocess_session_numberID") 
    test_normal_loader = generate('openstack_normal2.log_preprocess_session_numberID')
    test_abnormal_loader = generate('openstack_abnormal.log_preprocess_session_numberID')

python LogKeyModel_train.py
python LogKeyModel_predict.py


