package com.example.anomalydetection;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class FileTransferAnomalyDetection {
    public static void main(String[] args) throws Exception {
        // Load data
        String filePath = "data/file_transfer_logs.csv"; // Your dataset path
        int batchSize = 32;
        DataSetIterator dataIterator = DataLoader.loadData(filePath, batchSize);

        // Build and train the model
        int inputSize = dataIterator.inputColumns();
        MultiLayerNetwork model = new MultiLayerNetwork(AutoencoderModel.buildModel(inputSize));
        model.init();
        AutoencoderTrainer.trainModel(model, dataIterator, 10); // Train for 10 epochs

        // Test for anomalies
        INDArray testInput = Nd4j.create(new double[]{/* Sample test input */}, new int[]{1, inputSize});
        double error = AnomalyDetector.calculateReconstructionError(model, testInput);
        double threshold = 0.05; // Set based on validation data
        System.out.println("Reconstruction Error: " + error);
        System.out.println("Is Anomalous: " + AnomalyDetector.isAnomalous(error, threshold));
    }
}
