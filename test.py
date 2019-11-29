import json
import os
import sys
from multiprocessing import Process
# cluster_description = { "worker": ["192.168.11.113:3745", "192.168.11.162:3746"] }
cluster_description = { "worker": ["192.168.11.115:3745"] }
# os.environ["TF_CONFIG"] = json.dumps({ "cluster": { "worker": ["192.168.11.113:3745", "192.168.11.162:3746"] }, "task": {"type": "worker", "index": 0} })
os.environ["TF_CONFIG"] = json.dumps({ "cluster": { "worker": ["192.168.11.113:3745"] }, "task": {"type": "worker", "index": 0} })
import tensorflow as tf
cluster = tf.train.ClusterSpec(cluster_description)
import logging
logger = tf.get_logger()
logger.setLevel(logging.DEBUG)

def test_dist(task_id):
    server = tf.distribute.Server(
        cluster, job_name="worker", task_index=task_id)

    dist = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    print('num replicas', dist.num_replicas_in_sync)

    with tf.device('/job:worker/task:{0}/device:CPU:0'.format(task_id)):
        t = tf.Variable([1.0,3.0*task_id], dtype=tf.float32, name='myvar')

    def sum_deltas_fn(v):
        return tf.identity(v)

    with dist.scope():
        all_ts = dist.experimental_run_v2(sum_deltas_fn, args=[t])
        delta_sums_results = dist.reduce(tf.distribute.ReduceOp.SUM, all_ts)

        sess = tf.compat.v1.Session(server.target)
        sess.run(tf.compat.v1.global_variables_initializer())

        print('tensor', delta_sums_results)
        print('tensor value', sess.run(delta_sums_results))

test_dist(int(sys.argv[1]))

