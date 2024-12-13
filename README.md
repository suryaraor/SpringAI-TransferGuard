# SpringAI-TransferGuard

## AI-Powered Anomaly Detection for Secure File Transfers using Spring Boot and Deeplearning4j

## Overview
This repository provides a reference implementation of an anomaly detection system for file transfers, built using **Spring Boot** and **Deeplearning4j**. The application leverages an autoencoder neural network to detect anomalies in file transfer data based on reconstruction errors.

## Features
- **Autoencoder-based Anomaly Detection**: Learns normal patterns in file transfer logs to identify deviations.
- **Customizable Configuration**: Batch size, threshold, and other parameters can be adjusted.
- **Java-based Implementation**: Fully implemented in Java using Spring Boot and Deeplearning4j.

## Getting Started

### Prerequisites
- **Java Development Kit (JDK)** 11 or higher
- **Maven**
- **Python (Optional)**: For preprocessing datasets if required
- A CSV file containing file transfer logs with numerical features

### Clone the Repository
```bash
git clone https://github.com/suryaraor/SpringAI-TransferGuard.git
cd SpringAI-TransferGuard
```

### Build the Project
Run the following command to build the project:
```bash
mvn clean install
```

### Run the Application
1. Place your dataset (`file_transfer_logs.csv`) in the `data` directory.
2. Start the application:
```bash
mvn spring-boot:run
```

### Test the Application
The application automatically trains the autoencoder on the dataset and evaluates test samples for anomalies. Check the console output for:
- Reconstruction error
- Anomaly detection result

## Project Structure
```plaintext
src
├── main
│   ├── java
│   │   └── com
│   │       └── example
│   │           └── anomalydetection
│   │               ├── AnomalyDetectionApplication.java
│   │               ├── DataLoader.java
│   │               ├── AutoencoderModel.java
│   │               ├── AutoencoderTrainer.java
│   │               ├── AnomalyDetector.java
│   │               └── FileTransferAnomalyDetection.java
│   └── resources
│       └── application.properties
└── test
```

## Configuration
Edit `application.properties` for custom configurations:
```properties
# Custom properties can be added here
```

## Dataset Format
The input dataset should be a CSV file with numerical features representing:
- File size
- Transfer time
- Transfer status

Example:
```csv
1024,30,1
2048,45,0
```

## Improving the Codebase
Here are ways to enhance the application:
1. **Hyperparameter Tuning**: Adjust the number of epochs, learning rate, and network layers.
2. **Real-time Integration**: Stream file transfer data in real-time for anomaly detection.
3. **Dashboard**: Add a front-end dashboard for monitoring anomalies.
4. **Advanced Anomaly Models**: Integrate more sophisticated models like Variational Autoencoders (VAEs).

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Added feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Create a Pull Request.

## License
This project is licensed under the GNU General Public License v3.0.

## Contact
For any questions, feel free to reach out at **suryarao.r@utexas.edu**.
