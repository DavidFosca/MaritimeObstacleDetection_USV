# MaritimeObstacleDetection_USV
This work proposes an end to end solution for obstacle detection and proximity alert application for USV vehicles which require an extra layer of on-board intelligent navigation. 

The NTNU FishOtter USV is a mobile platform for autonomous robotic search that aims to facilitate investigation and understanding of fundamental biological processes in the ocean such as the migration and movement ecology of fishes. However, the FishOtter is not yet a fully autonomous vehicle. In addition, several field experiences have shown the threat that unknown small to medium size obstacles represent to the FishOtter. Therefore, it is imperative to make the USV self-aware of its surroundings during runtime so it can execute anti-collision maneuvers accordingly. Few studies and practical work exists around maritime obstacle detection, and several of the existing ones contemplate the use of LIDAR or radar sensors which are not suitable for small size USVs such as the FishOtter.

This work shows in detail an end-to-end development of an obstacle detection and proximity alert application for collision avoidance at sea using a stereo camera-based solution. From the design, integration, deployment and testing of the solution on real working conditions, this work represents a base to build a more robust system considering all the optimization potentials discovered along the way and mentioned in the previous chapter. In addition, the report condensates in technical detail the important steps from design to deployment of a deep learning model on an edge device such as the Jetson Nano and Xavier NX, and provides a series of source files well detailed to build such an application, in the GitHub repository of this project.

The hardware selected for this project is a stereo-camera ZED2i and a Jetson Xavier-NX for deployment of the solution at the edge, both provided by StereoLabs with an industrial-grade enclosure design (IP66). On the other hand, the deep learning component is a U-Net segmentation neural network called the OtterNet, developed using Python and Tensorflow. In addition, one of the public available datasets for maritime obstacle segmentation is used to train, validate and test the model. However, this project also shows how to start building a customized dataset from real recordings of the FishOtter, which includes left frame images from the stereo-camera and annotations created using MATLAB. This dataset, of a total of 100 images and annotations, is used for testing the OtterNet, and should be used to re-train the network expecting increase in accuracy. Furthermore, the stereo-vision depth estimates and the OtterNet are integrated in an obstacle detection and proximity estimation C++ application. After configuration of the edge device, the application is also deployed on the Jetson Xavier-NX. Besides, the application is also deployed on a Jetson Nano to assess its performance on a lower-cost and lower-performance hardware.

As already mentioned, the U-Net segmentation model is evaluated with 100 images manually annotated, resulting in a f1-score of 0.83. In addition, the accuracy of the stereo-depth estimates is also assessed, showing a relative error no larger than 5% for distances no longer than 15 meters. Finally, considering the resource constraint nature of edge devices such s the Jetson Xavier-NX and Nano, performance tests are executed to evaluate the latency of the application as well as the memory utilization and power consumption during run-time. These tests are important to discover potential optimization opportunities and alculate the processing speed of theapplication, which is approximately 1 frame persecond and 2 frames per second for the Jetson Nano and Jetson Xavier-NX respectively. Overall, the developed obstacle detection and proximity alert application meets the functional and technical requirements defined during chapter three, and shows promising results to be potentially used as part of a collision avoidance protocol. It is important to mention that this project sets a technical base line for further optimizations, in a way that it details the pipeline that goes from design to implementation and deployment using the different platforms required to build the application. In addition, this document works both as a technical guide for code understanding and platform configuration, as well as providing an overall explanation of the project design and motivation.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Below is an explanation of what can be found in each of the directories. It is important to mention that all of the source files include detailed comments. In addition, all the build procedures for the Jetson devices are executed using CMAKE, so it is important to respect the file structure already given in the directories and that can be directly cloned. In addition, all of the file PATHs have to be changed depending on the system and device. Finally, every Python script was developed using Google Colab, so they can also be easily deployed in that development environment.

  **Obstacle Detection and Proximity Alert Application**
  (i) Jetson Xavier NX and Nano: Includes two directories with the necessary files to
  build the C++ application on the Jetson devices. The application needs the name
  of the verification video as an input argument. In addition, it uses the stereo-camera
  data stream as an input and runs for 200 frames before it stops. This parameter
  is hard-coded so the source code has to be modified when the application needs to
  run continuously. The application also generates a verification video.
  (ii) Windows Visual Studio: Includes the files needed to build the C++ application
  in the development environment using Windows Visual Studio 2022. This application uses
  a ".svo" video as input and is used for development purposes because it
  allows to check results and make adjustments on the application without the need
  of actually being on the field with real camera data stream. The application will
  run until it finishes processing all the ".svo" video frames and will also provide a
  verification video.

  **ONNX Configuration CMAKE**
  (i) ONNX-cmakeconfig: Includes the cmake configuration file for ONNX runtime in
      Jetpack which is [provided](https://github.com/microsoft/onnxruntime/issues/3124).
  
  **Record Video at the Edge**
  (i) Record SVO Video: Includes the files needed to build the C++ application to
      record 100 frames in ".svo" format. This is a modified code provided by [StereolabsSDK 
      examples](https://github.com/stereolabs/zed-examples/tree/master/svo%20recording/recording/cpp)
      and is used to record the 60 videos on the field, in order to
      generate the first instances of the Otter Dataset.
  (ii) Export SVO to AVI: Includes the files needed to build the C++ application to
      export a SVO video to AVI format. This is a modified code provided by [StereolabsSDK 
      examples](https://github.com/stereolabs/zed-examples/tree/master/svo%20recording/recording/cpp)
      and is used to get a ".avi" extension video that is the input the
      Video Generation Python script needs to create the OtterNet output mask video
      results.
  
  **OtterNet Training**
  (i) Dataset Modification - Script: Includes the python script that converts the
      public dataset to the binarized format as required by the OtterNet training script.
  (ii) OtterNet Training - Script: Includes the python script to train the U-Net neural
        network called the OtterNet.
  (iii) Convert Model to ONNX - Script: Includes the python script to convert a
        keras tensorflow model to ONNX format. The script also includes the quantization
        procedure.
  (iv) MaSTr1325 Modified Dataset: Includes a Google Drive link to all the 1325
       images and binarized target masks from the original public dataset.
  (v) Otter Dataset: Includes a Google Drive to all the 100 images and binarized target
      masks from the Otter Dataset. In addition, it also includes the remaining 200 images
      that are still to be labeled.

  **Experiments**
  (i) Tegrastats Analysis: Includes the python script that analyzes Tegrastats log files
      for both edge devices: Jetson Xavier-NX or Nano.
  (ii) Depth Extraction: Includes the files needed to build the C++ application to
      extract depth measurements from a given pixel coordinate with a ".svo" video format
      recording as the input in Visual Studio 2022.
      Results
  (i) Obstacle Detection and Proximity Alert verification videos: Includes a
      Google Drive link to 5 the verification video results from the Jetson Xavier-NX.
  (ii) OtterNet Output videos: Includes a Google Drive link to 5 of the videos generated 
      from the OtterNet obstacle segmentation task done in Python


