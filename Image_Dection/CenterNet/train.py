import tensorflow as tf
import time
import os
import shutil

from core.centernet import CenterNet
from core.postprocessing import PostProcessing
from data.dataloader import DetectionDataset, DataLoader
from configuration import Config


def print_model_summary(network):
    sample_inputs = tf.random.normal(shape=(Config.batch_size, Config.get_image_size()[0], Config.get_image_size()[1], Config.image_channels))
    sample_outputs = network(sample_inputs, training=True)
    network.summary()
    print("inputs.shape:",sample_inputs.shape,"outputs.shape:",sample_outputs.shape)

def train():
    #开启GPU
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
    # 读取数据
    dataloader = DetectionDataset()
    train_data, train_size = dataloader.generate_datatset(mode="train")  #获取txt文件中的string数据，返回的是batch_size的string，以及txt的大小
    train_steps_per_epoch = tf.math.ceil(train_size / Config.batch_size)
    #验证集
    val_data,val_size = dataloader.generate_datatset(mode="val")
    val_steps_per_epoch = tf.math.ceil(val_size/Config.batch_size)
    data_loader = DataLoader()  #创建一个class的大小数据加载

    if os.path.exists(Config.log_dir):
        # 清除summary目录下原有的东西
        shutil.rmtree(Config.log_dir)

    # 建立模型保存目录
    if not os.path.exists(os.path.split(Config.save_model_path)[0]):
        os.mkdir(os.path.split(Config.save_model_path)[0])

    print('Total on {}, train on {} samples, val on {} samples with batch size {}.'.format((train_size + val_size), train_size, val_size, Config.batch_size))
    # optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4,
                                                                 decay_steps=train_steps_per_epoch * Config.learning_rate_decay_epochs,
                                                                 decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # 创建模型结构
    centernet = CenterNet()
    print_model_summary(centernet)
    try:
        centernet.load_weights(filepath=Config.save_model_path)
        print("Load weights...")
    except:
        print("load weights...")

    # 定义模型评估指标
    train_loss = tf.metrics.Mean(name='train_loss')
    valid_loss = tf.metrics.Mean(name='valid_loss')

    post_process = PostProcessing()
    #设置保存最好模型的指标
    best_test_loss = float('inf')

    # 创建summary
    summary_writer = tf.summary.create_file_writer(logdir=Config.log_dir)


    #训练
    for epoch in range(1,Config.epochs + 1):
        train_loss.reset_states()
        valid_loss.reset_states()

        #处理训练集数据
        for step, batch_data in enumerate(train_data):
            step_start_time = time.time()
            images, labels = data_loader.read_batch_data(batch_data)  # 返回的是图片image,以及标签信息[batch, max_boxes_per_image, xmin, ymin, xmax, ymax, class_id]
            with tf.GradientTape() as tape:
                # 得到预测
                pred = centernet(images, training=True)
                # 计算损失
                loss_value = post_process.training_procedure(batch_labels=labels, pred=pred)

            # 反向传播梯度下降
            # model.trainable_variables代表把loss反向传播到每个可以训练的变量中
            gradients = tape.gradient(target=loss_value, sources=centernet.trainable_variables)
            # 将每个节点的误差梯度gradients，用于更新该节点的可训练变量值
            # zip是把梯度和可训练变量值打包成元组
            optimizer.apply_gradients(grads_and_vars=zip(gradients, centernet.trainable_variables))

            # 更新train_loss
            train_loss.update_state(values=loss_value)

            step_end_time = time.time()
            print("Epoch: {}/{}, step: {}/{}, loss: {}, time_cost: {:.3f}s".format(epoch,
                                                                                  Config.epochs,
                                                                                  step,
                                                                                  train_steps_per_epoch,
                                                                                  train_loss.result(),
                                                                                  step_end_time - step_start_time))

            with summary_writer.as_default():
                tf.summary.scalar("steps_perbatch_train_loss", train_loss.result(), step=tf.cast(((epoch - 1) * train_steps_per_epoch + step),tf.int64))

        # 计算验证集
        for step, batch_data in enumerate(val_data):
            step_start_time = time.time()
            images, labels = data_loader.read_batch_data(batch_data)  # 返回的是图片image,以及标签信息[batch, max_boxes_per_image, xmin, ymin, xmax, ymax, class_id]
            # 得到预测，不training
            pred = centernet(images)
            # 计算损失
            loss_value = post_process.training_procedure(batch_labels=labels, pred=pred)

            # 更新valid_loss
            valid_loss.update_state(loss_value)
            step_end_time = time.time()
            print("--------Epoch: {}/{}, step: {}/{}, loss: {}, time_cost: {:.3f}s".format(epoch,
                                                                                  Config.epochs,
                                                                                  step,
                                                                                  val_steps_per_epoch,
                                                                                  valid_loss.result(),
                                                                                  step_end_time - step_start_time))
            with summary_writer.as_default():
                tf.summary.scalar("steps_perbatch_val_loss", valid_loss.result(), step=tf.cast((epoch - 1) * val_steps_per_epoch + step,tf.int64))

        # 保存到tensorboard里
        with summary_writer.as_default():
            tf.summary.scalar("train_loss",train_loss.result(), step=optimizer.iterations)
            tf.summary.scalar('valid_loss', valid_loss.result(), step=optimizer.iterations)

        # 只保存最好模型
        if valid_loss.result() < best_test_loss:
            best_test_loss = valid_loss.result()
            centernet.save_weights(Config.save_model_path,save_format="tf")
            print("Update model's weights")

if __name__ == '__main__':
    print("Code starts to run.....")
    train()
    print("Code is running at the end.")