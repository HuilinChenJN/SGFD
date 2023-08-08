import tensorflow as tf
import numpy as np

class TeacherModel(tf.keras.Model):
    def __init__(self, inputs, outputs, is_pruning=True, **kwargs):
        super(TeacherModel, self).__init__(**kwargs)
        self.feature_dim = inputs
        self.dim_latent = outputs
        self.decay_ration = 2
        self.is_pruning = is_pruning
        self.transfer_multilayer = tf.keras.Sequential()
        liner_feat_dim = self.feature_dim
        while True:
            liner_feat_dim = int(liner_feat_dim / self.decay_ration)
            if liner_feat_dim < self.dim_latent:
                self.transfer_multilayer.add(tf.keras.layers.Dense(self.dim_latent))
                break
            else:
                self.transfer_multilayer.add(tf.keras.layers.Dense(liner_feat_dim, input_dim=liner_feat_dim * self.decay_ration))
                if liner_feat_dim == self.dim_latent:
                    break

    def __call__(self, inputs):

        out = self.transfer_multilayer(inputs)
        if self.is_pruning:
            out = tf.nn.leaky_relu(out)

        return out


class StudentModel(tf.keras.Model):
    def __init__(self, inputs, outputs, is_pruning=True):
        super(StudentModel, self).__init__(name='studentModel')
        self.feature_dim = inputs
        self.dim_latent = outputs
        self.is_pruning = is_pruning
        # inputs=hidden_layer, units=1 * self.embed_dim
        self.MLP = tf.keras.layers.Dense(self.dim_latent)

    def __call__(self, inputs):

        out = self.MLP(inputs)
        if self.is_pruning:
            out = tf.nn.leaky_relu(out)

        return out



if __name__ == '__main__':
    student_model = StudentModel(4096, 128, name='studentModel')
    teacher_model = TeacherModel(4096, 128)
    feature = tf.random_normal(shape=(3000,4096))

    student_feature = student_model(feature)
    teacher_feature = teacher_model(feature)
    print(student_feature.shape, teacher_feature.shape)
    # print(student_feature)
