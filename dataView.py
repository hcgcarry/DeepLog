# This is just a data viewing to see there are how many templates, training data and so on.
if __name__ == '__main__':
    hdfs_train = []
    hdfs_test_normal = []
    hdfs_test_abnormal = []
    h1 = set()
    h2 = set()
    h3 = set()
    train_log = '/workspace/anomaly_detecton/DeepLog/data/query.log_normal_afterPreProcess_session_numberID'
    test_log = '/workspace/anomaly_detecton/DeepLog/data/query.log_normal_afterPreProcess_session_numberID'
    test_abnormal_log= '/workspace/anomaly_detecton/DeepLog/data/query.log_abnormal_afterPreProcess_session_numberID'
    with open(train_log, 'r') as f:
        for line in f.readlines():
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            hdfs_train.append(line)
    for line in hdfs_train:
        for c in line:
            h1.add(c)

    with open(test_log, 'r') as f:
        for line in f.readlines():
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            hdfs_test_normal.append(line)
    for line in hdfs_test_normal:
        for c in line:
            h2.add(c)

    with open(test_abnormal_log, 'r') as f:
        for line in f.readlines():
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            hdfs_test_abnormal.append(line)
    for line in hdfs_test_abnormal:
        for c in line:
            h3.add(c)
    print('train file %s length: %d, template length: %d, template: %s' % (train_log ,len(hdfs_train), len(h1), h1))
    print('test_normal %s length: %d, template length: %d, template: %s' % (test_log, len(hdfs_test_normal), len(h2), h2))
    print('test_abnormal %s length: %d, template length: %d, template: %s' % (test_abnormal_log ,len(hdfs_test_abnormal), len(h3), h3))
