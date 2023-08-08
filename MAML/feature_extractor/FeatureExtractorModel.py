import tensorflow as tf
# import numpy as np
from feature_extractor.TeacherStudentModel import TeacherModel,StudentModel


class FeatureExtractorModel(tf.keras.Model):
    def __init__(self, feature, feature_dim, t_decay, meta_label, is_pruning=True, dim_latent=64):
        super(FeatureExtractorModel, self).__init__()
        self.meta_label = meta_label
        self.label_size = 25
        self.dim_latent = dim_latent
        self.is_pruning = is_pruning
        self.features = feature
        self.t = t_decay
        self.student_model = StudentModel(feature_dim, self.dim_latent, self.is_pruning)
        self.teacher_model = TeacherModel(feature_dim, self.dim_latent, self.is_pruning)

    def category_classification(self, input):
        return tf.layers.dense(inputs=input, units=self.label_size, reuse=tf.AUTO_REUSE, name="category_classification")

    def l2_loss(self, y_true, y_pred):
      squared_difference = tf.square(y_true - y_pred)
      return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

    def call(self, inputs):
        nodes = inputs
        node_feature = tf.gather(self.features, axis=0, indices=nodes) # self.features[nodes]
        node_label = tf.gather(self.meta_label, axis=0, indices=nodes) # self.meta_label[nodes]
        node_label = tf.one_hot(node_label, self.label_size)

        # 1. obtain the probability distribution of teacher model
        teacher_x = self.teacher_model(node_feature)
        teacher_result = self.category_classification(teacher_x)
        teacher_soft_result = tf.keras.activations.softmax(teacher_result / self.t, axis=1)
        teacher_hard_result = tf.keras.activations.softmax(teacher_result, axis=1)

        # 2. obtain the probability distribution of student model
        student_x = self.student_model(node_feature)
        student_result = self.category_classification(student_x)
        student_soft_result = tf.keras.activations.softmax(student_result / self.t, axis=-1)

        # 3. The teacher extractor is optimized using CE loss
        label_class_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=teacher_hard_result, labels=node_label)
        label_class_loss = tf.reduce_sum(label_class_loss)

        # 4. obtain the knowledge distillation based on response and feature
        label_kd_loss = self.l2_loss(teacher_soft_result, student_soft_result)
        feature_constraint_loss = self.l2_loss(student_x, teacher_x)

        return teacher_x, label_class_loss, label_kd_loss, feature_constraint_loss


if __name__ == '__main__':
    feature = tf.random_normal(shape=(3000, 4096))
    meta_label = tf.random_normal(shape=(3000, 25))
    # feature, feature_dim, meta_label, is_pruning, dim_latent=64
    feature_extrator = FeatureExtractorModel(feature, 4096, meta_label,True,dim_latent=128)
    feature_extrator.compile(optimizer="Adam")
    node_input = tf.constant([0,13,45,675,4,31,234,657])
    out = feature_extrator.fit(node_input, epochs=1, steps_per_epoch=8)
    print(out)