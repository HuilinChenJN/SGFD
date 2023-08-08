import tensorflow as tf

class FeatureFusionModel(tf.keras.Model):
    def __init__(self, meta_label, dim_latent):
        super(FeatureFusionModel, self).__init__()
        self.dim_latent = dim_latent
        self.meta_label = meta_label
        self.label_size = len(set(meta_label.tolist()))

    def feature_transfer_layer(self, input):
        return tf.layers.dense(inputs=input, units=self.dim_latent, name="feature_transfer_layer")

    def category_classification(self, input):
        return tf.layers.dense(inputs=input, units=self.label_size, name="teacher_fusion_category_classification")

    def __call__(self, nodes, fusion, has_n=True):
        gt_prediction = tf.gather(self.meta_label, axis=0, indices=nodes)  # self.meta_label[nodes]
        gt_prediction = tf.one_hot(gt_prediction, self.label_size)

        transfer_layer = self.feature_transfer_layer(fusion)

        if has_n:
            transfer_layer = tf.keras.activations.relu(transfer_layer)

        result = self.category_classification(transfer_layer)

        class_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=gt_prediction)
        class_loss = tf.reduce_sum(class_loss)

        return class_loss

