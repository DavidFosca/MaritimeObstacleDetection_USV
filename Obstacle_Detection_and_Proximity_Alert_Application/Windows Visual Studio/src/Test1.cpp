#include <sl/Camera.hpp>
#include <iostream>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include "Helpers.cpp"

using namespace cv;
using namespace sl;
using namespace std;
using namespace std::chrono;

int main(int argc, char** argv)
{   
    if (argc < 2) {
        cout << "pass only the name of verification video including extension (.avi).\n";
        return EXIT_FAILURE;
    }

    //Timestamps variables.
    auto start_inference = high_resolution_clock::now();
    auto duration_inference = duration_cast<microseconds>(high_resolution_clock::now() - start_inference);
    auto start_proximity_alert = high_resolution_clock::now();
    auto duration_proximity = duration_cast<microseconds>(high_resolution_clock::now() - start_proximity_alert);
    auto start_execution_time = high_resolution_clock::now();
    auto total_execution_time = duration_cast<microseconds>(high_resolution_clock::now() - start_proximity_alert);
    auto start_get_frame = high_resolution_clock::now();
    auto duration_get_frame = duration_cast<microseconds>(high_resolution_clock::now() - start_proximity_alert);
    auto start_get_depth = high_resolution_clock::now();
    auto duration_get_depth = duration_cast<microseconds>(high_resolution_clock::now() - start_proximity_alert);
    int mean_duration_inference = 0;
    int mean_duration_proximity = 0;
    int mean_duration_frame = 0;
    int mean_duration_depth = 0;
    start_execution_time = high_resolution_clock::now();

    // ------------------------- Onnxruntime initialization ------------------------- //
    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::Session session(nullptr);
    //Define Input and Output dimension (W,H,C).
    constexpr int64_t width = 256;
    constexpr int64_t height = 256;
    constexpr int64_t numChannels = 3;
    constexpr int64_t numClasses = 1;
    //Define input and output vector-flatten size.
    constexpr int64_t numOutputClasses = numClasses * height * width;
    constexpr int64_t numInputElements = numChannels * height * width;
    //Define path from where onnx model will be loaded.
    const wchar_t* modelPath = L"\\home\\user\\Desktop\\NTNU\\Cybernetic Project\\ONNX_C++\\unet_mobilnet_opset12.onnx";
    //Create ONNX Session with OtterNet model.
    session = Ort::Session(env, modelPath, Ort::SessionOptions{ nullptr });
    //Define and initialize the Input and Output Tensors.
    const array<int64_t, 4> inputShape = { 1, height, width, numChannels };
    const array<int64_t, 4> outputShape = { 1, height, width, numClasses };
    float* input = new float[numInputElements];
    float* results = new float[numOutputClasses];
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input, numInputElements, inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results, numOutputClasses, outputShape.data(), outputShape.size());
    //Define Input and Output names from the OtterNet.
    Ort::AllocatorWithDefaultOptions ort_alloc;
    char* inputName = session.GetInputName(0, ort_alloc);
    char* outputName = session.GetOutputName(0, ort_alloc);
    const array<const char*, 1> inputNames = { inputName };
    const array<const char*, 1> outputNames = { outputName };

    // ------------------------- ZED-SDK initialization ------------------------- //
    const int frameWidth = 2208;
    const int frameHeigth = 1242;
    //Input SVO video.
    const char* video_svo = "\\home\\user\\Desktop\\App\\videos\\videos_input\\record_22.svo";
    //Define variables for outputvideo generation.
    const std::string write_path = "\\home\\user\\Desktop\\App\\videos\\imgOut_svo";
    std::string ver_video_name(argv[1]);
    const std::string write_video_path_1 = "\\home\\user\\Desktop\\App\\videos\\verification_videos\\";
    const std::string write_video_path = write_video_path_1 + ver_video_name;
    //Create video writer object
    cv::VideoWriter output(write_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frameWidth, frameHeigth));
    std::string index;
    //Create zed camera object and parameters.
    Camera zed;
    InitParameters init_parameters;
    init_parameters.input.setFromSVOFile(video_svo);
    //Set configuration parameters
    init_parameters.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
    //Set depth measurements in millimeters.
    init_parameters.coordinate_units = UNIT::MILLIMETER;
    //Configure camera object with defined parameters from video.
    auto returned_state = zed.open(init_parameters);
    auto resolution = zed.getCameraInformation().camera_configuration.resolution;
    //Define OpenCV image window size 2208x1242.
    sl::Resolution low_resolution(min(frameWidth, (int)resolution.width), min(frameHeigth, (int)resolution.height));
    sl::Mat svo_image(low_resolution, MAT_TYPE::U8_C4, MEM::CPU);
    cv::Mat svo_image_ocv = slMat2cvMat(svo_image);
    //Print Number of Frames in SVO Video.
    int svo_frame_rate = zed.getInitParameters().camera_fps;
    int nb_frames = zed.getSVONumberOfFrames();
    cerr << "SVO # frames = " << endl << " " << to_string(nb_frames) << endl << endl;
    //Setup counters, images, and timestamps variables.
    cv::Mat img_result;
    sl::Mat depth_map;
    float depth_value = 0;
    int count_1mtr = 0;
    int count_3mtr = 0;
    int count_5mtr = 0;
    int count_10mtr = 0;
    int count_20mtr = 0;
    int alert_1mtr = 0;
    int alert_3mtr = 0;
    int alert_5mtr = 0;
    int alert_10mtr = 0;
    int alert_20mtr = 0;
    //Define Threshold to evaluate if obstacle or not.
    int threshold_tobe_object = 50;
    //Define number of frames required to evaluate condition of proximity alarm.
    const int n_frames = 5;
    //Define the number of Frames that will be processed. ONLY for trials, it can be removed for continuous functioning.
    int nb_iterations = 200;
    cout << "duration_initialization: " << duration_cast<microseconds>(high_resolution_clock::now() - start_execution_time).count() << endl << endl;
    
    // ------------------------- Obstacle Detection and Proximity Alert Execution ------------------------- //
    int finish_frames_read = 0;
    int n_iter = 0;
    int i = 0;
    while (finish_frames_read == 0) {
        count_1mtr = 0;
        count_3mtr = 0;
        count_5mtr = 0;
        count_10mtr = 0;
        count_20mtr = 0;
        start_get_frame = high_resolution_clock::now();
        returned_state = zed.grab();
        if (returned_state == ERROR_CODE::SUCCESS) {
            //Get frame from camera.
            zed.retrieveImage(svo_image, VIEW::LEFT, MEM::CPU, low_resolution);
            //Process frame.
            vector<float> imageVec = loadImage(svo_image_ocv);
            //Copy image data to tensor input array.
            copy(imageVec.begin(), imageVec.end(), input);
            duration_get_frame = duration_cast<microseconds>(duration_get_frame + duration_cast<microseconds>(high_resolution_clock::now() - start_get_frame));
            //Retrieve depth map.
            start_get_depth = high_resolution_clock::now();
            zed.retrieveMeasure(depth_map, MEASURE::DEPTH);
            duration_get_depth = duration_cast<microseconds>(duration_get_depth + duration_cast<microseconds>(high_resolution_clock::now() - start_get_depth));
            // ------------------------- Run OtterNet Inference ------------------------- //
            start_inference = high_resolution_clock::now();
            //Run Inference.
            try {
                session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
            }
            catch (Ort::Exception& e) {
                cout << e.what() << endl;
                return 1;
            }
            // ------------------------- Evaluate Proximity of Obstacle ------------------------- //
            duration_inference = duration_cast<microseconds>(duration_inference + duration_cast<microseconds>(high_resolution_clock::now() - start_inference));
            start_proximity_alert = high_resolution_clock::now();
            //Convert output mask so that pixels classified as not obstacles (value = 1) are colored as white (255 pixel value).
            //This is only relevant to include when it is desired to have a verification video.
            for (size_t col = 0; col < numOutputClasses; col++) {
                if (results[col] > 0.5)
                {
                    results[col] = 255;
                }
                else
                {
                    //Obstacle is 0 (zero)
                    results[col] = 0;
                }
            }

            index = write_path + ".png";
            //Convert from flatten output to 2D image dimensions.
            img_result = cv::Mat(256, 256, CV_32FC1, results);
            //Due to the fact the the output target is 256x256, it is necessary to resize it back to the original stereo-camera frame 
            //resolution because the depth map is calculated for that frame size.
            cv::resize(img_result, img_result, Size(frameWidth, frameHeigth));
            cv::imwrite(index, img_result);
            cv::Mat img_result = imread(index, IMREAD_GRAYSCALE);
            //Due to the fact the the output target is 256x256, it is necessary to resize it back to the original stereo-camera frame 
            for (size_t y = 0; y < img_result.rows; y++) {
                for (size_t x = 0; x < img_result.cols; x++) {
                    //If pixel is classified as obstacle, then get its depth.
                    if (img_result.at<uchar>(Point(x, y)) != 255)
                    {
                        //For the verification video, every pixel classified as an obstacle that is not in the proximity intervals, is 
                        //given a 240 value.
                        img_result.at<uchar>(Point(x, y)) = 240;
                        //Calculate depth map.
                        depth_map.getValue(x, y, &depth_value);
                        //Proximity Interval: 1m.
                        if (depth_value <= 1000) {
                            img_result.at<uchar>(Point(x, y)) = 0;
                            count_1mtr++;
                        }
                        //Proximity Interval: 1m - 2m.
                        else if ((depth_value > 1000) && (depth_value <= 2000)) {
                            img_result.at<uchar>(Point(x, y)) = 50;
                            count_3mtr++;
                        }
                        //Proximity Interval: 2m - 3m.
                        else if ((depth_value > 2000) && (depth_value <= 3000)) {
                            img_result.at<uchar>(Point(x, y)) = 100;
                            count_5mtr++;
                        }
                        //Proximity Interval: 3m - 4m.
                        else if ((depth_value > 3000) && (depth_value <= 4000)) {
                            img_result.at<uchar>(Point(x, y)) = 100;
                            count_10mtr++;
                        }
                        //Proximity Interval: 4m - 5m.
                        else if ((depth_value > 4000) && (depth_value <= 5000)) {
                            img_result.at<uchar>(Point(x, y)) = 150;
                            count_20mtr++;
                        }
                    }
                }
            }

            //This step is needed to save the fram in video.
            cvtColor(img_result, img_result, COLOR_GRAY2BGR);
            //Write the alert messages in the image.
            cv::putText(img_result, //target image
                "- Object approaching (< 1 meter): ", //text
                cv::Point(10, 30), //top-left position
                cv::FONT_HERSHEY_DUPLEX,
                1.0,
                CV_RGB(62, 2, 175), //font color
                2);
            cv::putText(img_result, //target image
                "- Object approaching (1 to 2 meters):", //text
                cv::Point(10, 80), //top-left position
                cv::FONT_HERSHEY_DUPLEX,
                1.0,
                CV_RGB(62, 2, 175), //font color
                2);
            cv::putText(img_result, //target image
                "- Object approaching (2 to 3 meters):", //text
                cv::Point(10, 130), //top-left position
                cv::FONT_HERSHEY_DUPLEX,
                1.0,
                CV_RGB(62, 2, 175), //font color
                2);
            cv::putText(img_result, //target image
                "- Object approaching (3 to 4 meters):", //text
                cv::Point(10, 180), //top-left position
                cv::FONT_HERSHEY_DUPLEX,
                1.0,
                CV_RGB(62, 2, 175), //font color
                2);
            cv::putText(img_result, //target image
                "- Object approaching (4 to 5 meters):", //text
                cv::Point(10, 230), //top-left position
                cv::FONT_HERSHEY_DUPLEX,
                1.0,
                CV_RGB(62, 2, 175), //font color
                2);

            if (count_1mtr >= threshold_tobe_object) {
                cv::putText(img_result, //target image
                    " Alert!", //text
                    cv::Point(700, 30), //top-left position
                    cv::FONT_HERSHEY_DUPLEX,
                    1.0,
                    CV_RGB(189, 34, 1), //font color
                    2);
            }
            else {
                cv::putText(img_result, //target image
                    " No threat...", //text
                    cv::Point(700, 30), //top-left position
                    cv::FONT_HERSHEY_DUPLEX,
                    1.0,
                    CV_RGB(7, 180, 72), //font color
                    2);
            }
            if (count_3mtr >= threshold_tobe_object) {
                cv::putText(img_result, //target image
                    " Alert!", //text
                    cv::Point(700, 80), //top-left position
                    cv::FONT_HERSHEY_DUPLEX,
                    1.0,
                    CV_RGB(189, 34, 1), //font color
                    2);
            }
            else {
                cv::putText(img_result, //target image
                    " No threat...", //text
                    cv::Point(700, 80), //top-left position
                    cv::FONT_HERSHEY_DUPLEX,
                    1.0,
                    CV_RGB(7, 180, 72), //font color
                    2);
            }
            if (count_5mtr >= threshold_tobe_object) {
                cv::putText(img_result, //target image
                    " Alert!", //text
                    cv::Point(700, 130), //top-left position
                    cv::FONT_HERSHEY_DUPLEX,
                    1.0,
                    CV_RGB(189, 34, 1), //font color
                    2);
            }
            else {
                cv::putText(img_result, //target image
                    " No threat...", //text
                    cv::Point(700, 130), //top-left position
                    cv::FONT_HERSHEY_DUPLEX,
                    1.0,
                    CV_RGB(7, 180, 72), //font color
                    2);
            }
            if (count_10mtr >= threshold_tobe_object) {
                cv::putText(img_result, //target image
                    " Alert!", //text
                    cv::Point(700, 180), //top-left position
                    cv::FONT_HERSHEY_DUPLEX,
                    1.0,
                    CV_RGB(189, 34, 1), //font color
                    2);
            }
            else {
                cv::putText(img_result, //target image
                    " No threat...", //text
                    cv::Point(700, 180), //top-left position
                    cv::FONT_HERSHEY_DUPLEX,
                    1.0,
                    CV_RGB(7, 180, 72), //font color
                    2);
            }
            if (count_20mtr >= threshold_tobe_object) {
                cv::putText(img_result, //target image
                    " Alert!", //text
                    cv::Point(700, 230), //top-left position
                    cv::FONT_HERSHEY_DUPLEX,
                    1.0,
                    CV_RGB(189, 34, 1), //font color
                    2);
            }
            else {
                cv::putText(img_result, //target image
                    " No threat...", //text
                    cv::Point(700, 230), //top-left position
                    cv::FONT_HERSHEY_DUPLEX,
                    1.0,
                    CV_RGB(7, 180, 72), //font color
                    2);
            }
            //Write image into video. 
            output.write(img_result);

            // ------------------------- Send Proximity Alert ------------------------- //
            if (n_iter == n_frames) {
                if (alert_1mtr >= n_frames - 1) {
                    //Print alert in the terminal for user to see. Here is where the alert message can be send through IMC to the Otter Controller.
                    cout << "-Alert! Object less than 1 meter." << endl;
                }
                if (alert_3mtr >= n_frames - 1) {
                    //Print alert in the terminal for user to see.
                    cout << "--Alert! Object between 1 and 2 meters." << endl;
                }
                if (alert_5mtr >= n_frames - 1) {
                    //Print alert in the terminal for user to see.
                    cout << "---Alert! Object between 2 and 3 meters." << endl;
                }
                if (alert_10mtr >= n_frames - 1) {
                    //Print alert in the terminal for user to see.
                    cout << "----Alert! Object between 3 and 4 meters." << endl;
                }
                if (alert_20mtr >= n_frames - 1) {
                    //Print alert in the terminal for user to see.
                    cout << "-----Alert! Object between 4 and 5 meters." << endl;
                }
                n_iter = 0;
                alert_3mtr = 0;
                alert_1mtr = 0;
                alert_5mtr = 0;
                alert_10mtr = 0;
                alert_20mtr = 0;
            }
            else
            {
                //If the number of obstacle pixels is more or equal to the threshold of what is considered as an obstacle, then increment counter.
                if (count_1mtr >= threshold_tobe_object) {
                    alert_1mtr++;
                }
                if (count_3mtr >= threshold_tobe_object) {
                    alert_3mtr++;
                }
                if (count_5mtr >= threshold_tobe_object) {
                    alert_5mtr++;
                }
                if (count_10mtr >= threshold_tobe_object) {
                    alert_10mtr++;
                }
                if (count_20mtr >= threshold_tobe_object) {
                    alert_20mtr++;
                }
                n_iter++;
            }
            duration_proximity = duration_cast<microseconds>(duration_proximity + duration_cast<microseconds>(high_resolution_clock::now() - start_proximity_alert));
            i++;
        }
        else if (returned_state == sl::ERROR_CODE::END_OF_SVOFILE_REACHED)
        {
            finish_frames_read = 1;
        }
        else {
            cout << "Grab ZED: " << endl << " " << returned_state << endl << endl;
            break;
        }
    }
    //Calculate the response time of each section and print them at the terminal.
    mean_duration_inference = duration_inference.count() / i;
    mean_duration_proximity = duration_proximity.count() / i;
    mean_duration_frame = duration_get_frame.count() / i;
    mean_duration_depth = duration_get_depth.count() / i;
    total_execution_time = duration_cast<microseconds>(high_resolution_clock::now() - start_execution_time);
    cout << "mean_duration_inference: " << mean_duration_inference << endl << endl;
    cout << "mean_duration_proximity: " << mean_duration_proximity << endl << endl;
    cout << "mean_duration_frame: " << mean_duration_frame << endl << endl;
    cout << "mean_duration_depth: " << mean_duration_depth << endl << endl;
    cout << "total_executio_time: " << total_execution_time.count() << endl << endl;
    output.release();
    zed.close();
}