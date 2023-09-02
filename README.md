The extensive ubiquitous availability of sensors in smart devices and the Internet of Things (IoT) has opened up the possibilities for implementing sensor-based activity recognition. As opposed to traditional sensor time-series processing and hand- engineered feature extraction, in light of deep learning's proven effectiveness across various domains, numerous deep methods have been explored to tackle the challenges in activity recognition, outperforming the traditional signal processing and traditional machine learning approaches. In this work, by performing extensive experimental studies on two large-scale human activity recognition datasets, we investigate the performance of common deep learning and machine learning approaches as well as different training mechanisms (such as contrastive learning), and various feature representations extracted from the sensor time series data and measure their effectiveness for the human activity recognition task.

Read Readme file in the Data_ folder first.

Create a virtual env and install these packages:

    pip install torch numpy scikit-learn matplotlib pandas

After that run the main.py 3 (for run the model on WISDM dataset) or main.py 6 (for run the model on Meta_Har dataset)

    python main.py 3

or

    python main.py 6

