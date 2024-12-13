package com.example.anomalydetection;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

public class AnomalyDetector {
    public static double calculateReconstructionError(MultiLayerNetwork model, INDArray input) {
        INDArray reconstructed = model.output(input, false);
        INDArray diff = input.sub(reconstructed);
        return diff.norm2Number().doubleValue(); // L2 Norm as error metric
    }

    public static boolean isAnomalous(double error, double threshold) {
        return error > threshold; // Compare error to predefined threshold
    }
}
