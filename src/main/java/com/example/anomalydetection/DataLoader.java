package com.example.anomalydetection;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.impl.ListDataSetIterator;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class DataLoader {
    public static DataSetIterator loadData(String filePath, int batchSize) throws Exception {
        List<INDArray> data = new ArrayList<>();

        // Read CSV file
        Scanner scanner = new Scanner(new File(filePath));
        while (scanner.hasNextLine()) {
            String[] line = scanner.nextLine().split(",");
            double[] features = new double[line.length];
            for (int i = 0; i < line.length; i++) {
                features[i] = Double.parseDouble(line[i]);
            }
            data.add(Nd4j.create(features));
        }
        scanner.close();

        // Create a DataSet
        List<DataSet> dataSetList = new ArrayList<>();
        for (INDArray row : data) {
            dataSetList.add(new DataSet(row, row)); // Autoencoder: input equals output
        }
        return new ListDataSetIterator<>(dataSetList, batchSize);
    }
}
