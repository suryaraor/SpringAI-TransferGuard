package com.example.anomalydetection;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class AutoencoderTrainer {
    public static void trainModel(MultiLayerNetwork model, DataSetIterator data, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            data.reset();
            model.fit(data);
            System.out.println("Epoch " + epoch + " completed.");
        }
    }
}
