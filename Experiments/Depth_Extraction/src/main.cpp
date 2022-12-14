
// ZED include
#include <sl/Camera.hpp>
// Sample includes
#include <opencv2/opencv.hpp>
#include "utils.hpp"

#include <iostream>
//#include <onnxruntime_cxx_api.h>

#include "Helpers.cpp"

// Using namespace
using namespace sl;
using namespace std;

void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");

int main(int argc, char **argv) {

    //char *p1 = "./ZED_SVO_Playback C:/Users/David/Desktop/NTNU/Cybernetic Project/ZED/Videos_Test";
    argv[0] = "./ZED_SVO_Playback";
    argv[1] = "C:/Users/David/Desktop/Record depth/depth_test_7.svo";
    argc = 2;

    if (argc<=1)  {
        cout << "Usage: \n";
        cout << "$ ZED_SVO_Playback <SVO_file> \n";
        cout << "  ** SVO file is mandatory in the application ** \n\n";
        return EXIT_FAILURE;
    }

    // Create ZED objects
    Camera zed;
    InitParameters init_parameters;
    init_parameters.input.setFromSVOFile(argv[1]);
    init_parameters.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
    init_parameters.coordinate_units = UNIT::MILLIMETER; 

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    auto resolution = zed.getCameraInformation().camera_configuration.resolution;
    // Define OpenCV window size (resize to max 720/404)
    const int frameWidth = 2208;
    const int frameHeigth = 1242;
    sl::Resolution low_resolution(min(frameWidth, (int)resolution.width), min(frameHeigth, (int)resolution.height));
    sl::Mat svo_image(low_resolution, MAT_TYPE::U8_C4, MEM::CPU);
    cv::Mat svo_image_ocv = slMat2cvMat(svo_image);
    // Setup key, images, times
    char key = ' ';
    int svo_frame_rate = zed.getInitParameters().camera_fps;
    int nb_frames = zed.getSVONumberOfFrames();
    print("[Info] SVO contains " +to_string(nb_frames)+" frames");
    cv::Mat frame_array[547];
    sl::Mat depth_map;
    sl::Mat point_cloud;
    float depth_value = 0;

    // Start SVO playback
    int finish_frames_read = 0;
    int i = 0;
    while ((i <= 10) && (finish_frames_read==0)) {
        returned_state = zed.grab();
        if (returned_state == ERROR_CODE::SUCCESS) {

            // Get IMAGE
            zed.retrieveImage(svo_image, VIEW::LEFT, MEM::CPU, low_resolution);
            int svo_position = zed.getSVOPosition();
            // Display IMAGE
            frame_array[i] = svo_image_ocv;
            cout << "cv::M = " << endl << " " << frame_array[0].size() << endl << endl;
            //cv::imshow("View", frame_array[i]);
            //cv::waitKey(0);

            // Get DEPTH
            zed.retrieveMeasure(depth_map, MEASURE::DEPTH);
            // Calculate Depth of center point
            int x = 834;
            int y = 694;
            depth_map.getValue(x, y, &depth_value);
            printf("Frame Size: (%d , %d)\n", int(svo_image.getWidth()), int(svo_image.getHeight()));
            printf("Depth to Camera at (%d , %d): %f mm\n", x, y, depth_value);
            imwrite("C:/Users/David/Desktop/Record depth/Depth_Image_Frame/Frame_test_7.jpg", svo_image_ocv);
            // Get 3D POINT CLOUD
            //zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA);
            //point_cloud.write("C:/Users/David/Desktop/3DPointCloud/3Dtest.pcd");
            i++;
        }
        else if (returned_state == sl::ERROR_CODE::END_OF_SVOFILE_REACHED)
        {
            finish_frames_read = 1;
        }
        else {
            print("Grab ZED : ", returned_state);
            break;
        }
     } 
    
    zed.close();
    return EXIT_SUCCESS;
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
    cout <<"[Sample]";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    else
        cout<<" ";
    cout << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}
